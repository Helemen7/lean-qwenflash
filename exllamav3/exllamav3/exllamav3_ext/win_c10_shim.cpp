// Windows/ROCm only. torch's Windows build of c10 exports c10::ValueError's copy constructor but not the
// (SourceLocation, std::string) one inherited from c10::Error, and some torch headers (MultiStreamGuard via
// TORCH_CHECK_VALUE) reference it from cl-compiled units, which fails to link. Resolve the import with a
// stand-in that builds the base c10::Error; it only runs on a stream-mismatch check this extension never hits.
#if defined(_WIN32) && defined(USE_ROCM) && !defined(__clang__)

#include <c10/util/Exception.h>
#include <new>
#include <string>
#include <utility>

static void* valueerror_ctor_shim(void* self, c10::SourceLocation* loc, std::string* msg)
{
    new (self) c10::Error(*loc, std::move(*msg));
    return self;
}

extern "C" void* ex3_imp_valueerror_ctor = reinterpret_cast<void*>(&valueerror_ctor_shim);

#pragma comment(linker, "/alternatename:__imp_??0ValueError@c10@@QEAA@USourceLocation@1@V?$basic_string@DU?$char_traits@D@std@@V?$allocator@D@2@@std@@@Z=ex3_imp_valueerror_ctor")

#endif
