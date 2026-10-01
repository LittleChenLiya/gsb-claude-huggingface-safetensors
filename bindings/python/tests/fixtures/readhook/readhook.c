/*
 * readhook.dll - a small in-process read counter used by the
 * test_missing_keys.py test.
 *
 * It counts NtReadFile calls made by a *specific* module (the safetensors
 * rust extension, _safetensors_rust.pyd) by patching that module's import
 * address table (IAT) entry for ntdll!NtReadFile. Only the target module's
 * reads are counted; reads made by any other code (including Python itself)
 * are invisible to the counters.
 *
 * Why not use OS I/O counters? Some sandboxes virtualize
 * GetProcessIoCounters / page-fault counters to zero even for direct file
 * reads, so they cannot distinguish "read tensor bytes" from "only read the
 * header". This hook gives a deterministic in-process answer.
 *
 * The hook is installed automatically on load (DllMain) for the default
 * module name, and can be (re)installed on another module via
 * readhook_install(). Counters are process-global and reset with
 * readhook_reset().
 *
 * Build (x64, from a "x64 Native Tools Command Prompt for VS 2022"):
 *   cl /LD /O2 readhook.c
 */

#include <windows.h>
#include <string.h>

/* ntdll!NtReadFile signature (minimal, we only forward the call) */
typedef NTSTATUS (NTAPI *NtReadFile_t)(
    HANDLE FileHandle,
    HANDLE Event,
    PVOID  ApcRoutine,
    PVOID  ApcContext,
    PVOID  IoStatusBlock,
    PVOID  Buffer,
    ULONG  Length,
    PVOID  ByteOffset,
    PVOID  Key
);

#ifndef STATUS_NOT_IMPLEMENTED
#define STATUS_NOT_IMPLEMENTED ((NTSTATUS)0xC0000002L)
#endif

static volatile LONG64 g_read_ops = 0;
static volatile LONG64 g_read_bytes = 0;
static NtReadFile_t g_real_NtReadFile = NULL;

NTSTATUS NTAPI Hook_NtReadFile(
    HANDLE FileHandle,
    HANDLE Event,
    PVOID  ApcRoutine,
    PVOID  ApcContext,
    PVOID  IoStatusBlock,
    PVOID  Buffer,
    ULONG  Length,
    PVOID  ByteOffset,
    PVOID  Key
) {
    NtReadFile_t real = g_real_NtReadFile;
    InterlockedIncrement64(&g_read_ops);
    NTSTATUS status = real
        ? real(FileHandle, Event, ApcRoutine, ApcContext, IoStatusBlock,
               Buffer, Length, ByteOffset, Key)
        : STATUS_NOT_IMPLEMENTED;
    if (status == 0 /* STATUS_SUCCESS */ && IoStatusBlock) {
        /* IO_STATUS_BLOCK { NTSTATUS Status; ULONG_PTR Information; } */
        ULONG_PTR information = ((ULONG_PTR*)IoStatusBlock)[1];
        InterlockedAdd64(&g_read_bytes, (LONG64)information);
    }
    return status;
}

/* Patch every IAT entry named NtReadFile in the given module. */
static BOOL patch_module_iat(const wchar_t* module_name) {
    HMODULE mod = GetModuleHandleW(module_name);
    if (!mod) {
        return FALSE;
    }

    BYTE* base = (BYTE*)mod;
    IMAGE_DOS_HEADER* dos = (IMAGE_DOS_HEADER*)base;
    if (dos->e_magic != IMAGE_DOS_SIGNATURE) {
        return FALSE;
    }
    IMAGE_NT_HEADERS* nt = (IMAGE_NT_HEADERS*)(base + dos->e_lfanew);
    if (nt->Signature != IMAGE_NT_SIGNATURE) {
        return FALSE;
    }

    IMAGE_DATA_DIRECTORY imp_dir =
        nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    if (imp_dir.VirtualAddress == 0) {
        return FALSE;
    }

    BOOL patched_any = FALSE;
    IMAGE_IMPORT_DESCRIPTOR* desc =
        (IMAGE_IMPORT_DESCRIPTOR*)(base + imp_dir.VirtualAddress);
    for (; desc->Name; desc++) {
        IMAGE_THUNK_DATA* orig_thunk =
            (IMAGE_THUNK_DATA*)(base + desc->OriginalFirstThunk);
        IMAGE_THUNK_DATA* thunk =
            (IMAGE_THUNK_DATA*)(base + desc->FirstThunk);
        for (; orig_thunk->u1.AddressOfData; orig_thunk++, thunk++) {
            if (IMAGE_SNAP_BY_ORDINAL(orig_thunk->u1.Ordinal)) {
                continue;
            }
            IMAGE_IMPORT_BY_NAME* ibn =
                (IMAGE_IMPORT_BY_NAME*)(base + orig_thunk->u1.AddressOfData);
            if (strcmp((const char*)ibn->Name, "NtReadFile") != 0) {
                continue;
            }

            /* Save the relocated real pointer (first time only). */
            if (!g_real_NtReadFile) {
                g_real_NtReadFile = (NtReadFile_t)thunk->u1.Function;
            }

            DWORD old_protect = 0;
            if (VirtualProtect(&thunk->u1.Function, sizeof(ULONG_PTR),
                               PAGE_READWRITE, &old_protect)) {
                thunk->u1.Function = (ULONG_PTR)Hook_NtReadFile;
                DWORD tmp = 0;
                VirtualProtect(&thunk->u1.Function, sizeof(ULONG_PTR),
                               old_protect, &tmp);
                patched_any = TRUE;
            }
        }
    }

    if (!g_real_NtReadFile) {
        g_real_NtReadFile = (NtReadFile_t)GetProcAddress(
            GetModuleHandleW(L"ntdll.dll"), "NtReadFile");
    }
    return patched_any;
}

BOOL WINAPI DllMain(HINSTANCE instance, DWORD reason, LPVOID reserved) {
    (void)instance;
    (void)reserved;
    if (reason == DLL_PROCESS_ATTACH) {
        /* The rust extension module name. */
        patch_module_iat(L"_safetensors_rust.pyd");
    }
    return TRUE;
}

__declspec(dllexport) void readhook_reset(void) {
    InterlockedExchange64(&g_read_ops, 0);
    InterlockedExchange64(&g_read_bytes, 0);
}

__declspec(dllexport) LONG64 readhook_get_ops(void) {
    return g_read_ops;
}

__declspec(dllexport) LONG64 readhook_get_bytes(void) {
    return g_read_bytes;
}

__declspec(dllexport) LONG64 readhook_install(const wchar_t* module_name) {
    return patch_module_iat(module_name) ? 1 : 0;
}
