#include <Eigen/Core>
#include "query_cache.hpp"
#include "faer/internal/prologue.hpp"

#undef EIGEN_CPUID
#if !defined(EIGEN_NO_CPUID)
#if EIGEN_COMP_GNUC && EIGEN_ARCH_i386_OR_x86_64
#if defined(__PIC__) && EIGEN_ARCH_i386
// Case for x86 with PIC
#define EIGEN_CPUID(abcd, func, id)                                                                                    \
	__asm__ /* NOLINT */ __volatile__("xchgl %%ebx, %k1;cpuid; xchgl %%ebx,%k1"                                          \
	                                  : "=a"(abcd[0]), "=&r"(abcd[1]), "=c"(abcd[2]), "=d"(abcd[3])                      \
	                                  : "a"(func), "c"(id));
#elif defined(__PIC__) && EIGEN_ARCH_x86_64
// Case for x64 with PIC. In theory this is only a problem with recent gcc and with medium or large code model, not with
// the default small code model. However, we cannot detect which code model is used, and the xchg overhead is negligible
// anyway.
#define EIGEN_CPUID(abcd, func, id)                                                                                    \
	__asm__ /* NOLINT */ __volatile__("xchg{q}\t{%%}rbx, %q1; cpuid; xchg{q}\t{%%}rbx, %q1"                              \
	                                  : "=a"(abcd[0]), "=&r"(abcd[1]), "=c"(abcd[2]), "=d"(abcd[3])                      \
	                                  : "0"(func), "2"(id));
#else
// Case for x86_64 or x86 w/o PIC
#define EIGEN_CPUID(abcd, func, id)                                                                                    \
	__asm__ /* NOLINT */ __volatile__("cpuid"                                                                            \
	                                  : "=a"(abcd[0]), "=b"(abcd[1]), "=c"(abcd[2]), "=d"(abcd[3])                       \
	                                  : "0"(func), "2"(id));
#endif
#elif EIGEN_COMP_MSVC
#if (EIGEN_COMP_MSVC > 1500) && EIGEN_ARCH_i386_OR_x86_64
#define EIGEN_CPUID(abcd, func, id) __cpuidex((int*)abcd, func, id)
#endif
#endif
#endif

namespace fae {
namespace FAER_ABI_VERSION {
namespace internal {

#ifdef EIGEN_CPUID
auto eigen_cpuid_is_vendor(int const (&abcd)[4], int const (&vendor)[3]) -> bool {
	return abcd[1] == vendor[0] && abcd[3] == vendor[1] && abcd[2] == vendor[2];
}

void eigen_query_cache_intel_direct(int& l1, int& l2, int& l3) {
	int abcd[4];
	l1 = l2 = l3 = 0;
	int cache_id = 0;
	int cache_type = 0;
	do {
		abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
		EIGEN_CPUID(abcd, 0x4, cache_id);
		cache_type = (abcd[0] & 0x0F) >> 0;
		if (cache_type == 1 || cache_type == 3) // data or unified cache
		{
			int cache_level = (abcd[0] & 0xE0) >> 5;                 // A[7:5]
			int ways = int((unsigned(abcd[1]) & 0xFFC00000) >> 22U); // B[31:22]
			int partitions = (abcd[1] & 0x003FF000) >> 12;           // B[21:12]
			int line_size = (abcd[1] & 0x00000FFF) >> 0;             // B[11:0]
			int sets = (abcd[2]);                                    // C[31:0]

			int cache_size = (ways + 1) * (partitions + 1) * (line_size + 1) * (sets + 1);

			switch (cache_level) {
			case 1:
				l1 = cache_size;
				break;
			case 2:
				l2 = cache_size;
				break;
			case 3:
				l3 = cache_size;
				break;
			default:
				break;
			}
		}
		cache_id++;
	} while (cache_type > 0 && cache_id < 16);
}

void eigen_query_cache_intel_codes(int& l1, int& l2, int& l3) {
	int abcd[4];
	abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
	l1 = l2 = l3 = 0;
	EIGEN_CPUID(abcd, 0x00000002, 0);
	unsigned char* bytes = reinterpret_cast<unsigned char*>(abcd) + 2;
	bool check_for_p2_core2 = false;
	for (int i = 0; i < 14; ++i) {
		switch (bytes[i]) {
		case 0x0A:
			l1 = 8;
			break; // 0Ah   data L1 cache, 8 KB, 2 ways, 32 byte lines
		case 0x0C:
			l1 = 16;
			break; // 0Ch   data L1 cache, 16 KB, 4 ways, 32 byte lines
		case 0x0E:
			l1 = 24;
			break; // 0Eh   data L1 cache, 24 KB, 6 ways, 64 byte lines
		case 0x10:
			l1 = 16;
			break; // 10h   data L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
		case 0x15:
			l1 = 16;
			break; // 15h   code L1 cache, 16 KB, 4 ways, 32 byte lines (IA-64)
		case 0x2C:
			l1 = 32;
			break; // 2Ch   data L1 cache, 32 KB, 8 ways, 64 byte lines
		case 0x30:
			l1 = 32;
			break; // 30h   code L1 cache, 32 KB, 8 ways, 64 byte lines
		case 0x60:
			l1 = 16;
			break; // 60h   data L1 cache, 16 KB, 8 ways, 64 byte lines, sectored
		case 0x66:
			l1 = 8;
			break; // 66h   data L1 cache, 8 KB, 4 ways, 64 byte lines, sectored
		case 0x67:
			l1 = 16;
			break; // 67h   data L1 cache, 16 KB, 4 ways, 64 byte lines, sectored
		case 0x68:
			l1 = 32;
			break; // 68h   data L1 cache, 32 KB, 4 ways, 64 byte lines, sectored
		case 0x1A:
			l2 = 96;
			break; // code and data L2 cache, 96 KB, 6 ways, 64 byte lines (IA-64)
		case 0x22:
			l3 = 512;
			break; // code and data L3 cache, 512 KB, 4 ways (!), 64 byte lines, dual-sectored
		case 0x23:
			l3 = 1024;
			break; // code and data L3 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x25:
			l3 = 2048;
			break; // code and data L3 cache, 2048 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x29:
			l3 = 4096;
			break; // code and data L3 cache, 4096 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x39:
			l2 = 128;
			break; // code and data L2 cache, 128 KB, 4 ways, 64 byte lines, sectored
		case 0x3A:
			l2 = 192;
			break; // code and data L2 cache, 192 KB, 6 ways, 64 byte lines, sectored
		case 0x3B:
			l2 = 128;
			break; // code and data L2 cache, 128 KB, 2 ways, 64 byte lines, sectored
		case 0x3C:
			l2 = 256;
			break; // code and data L2 cache, 256 KB, 4 ways, 64 byte lines, sectored
		case 0x3D:
			l2 = 384;
			break; // code and data L2 cache, 384 KB, 6 ways, 64 byte lines, sectored
		case 0x3E:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 4 ways, 64 byte lines, sectored
		case 0x40:
			l2 = 0;
			break; // no integrated L2 cache (P6 core) or L3 cache (P4 core)
		case 0x41:
			l2 = 128;
			break; // code and data L2 cache, 128 KB, 4 ways, 32 byte lines
		case 0x42:
			l2 = 256;
			break; // code and data L2 cache, 256 KB, 4 ways, 32 byte lines
		case 0x43:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 4 ways, 32 byte lines
		case 0x44:
			l2 = 1024;
			break; // code and data L2 cache, 1024 KB, 4 ways, 32 byte lines
		case 0x45:
			l2 = 2048;
			break; // code and data L2 cache, 2048 KB, 4 ways, 32 byte lines
		case 0x46:
			l3 = 4096;
			break; // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines
		case 0x47:
			l3 = 8192;
			break; // code and data L3 cache, 8192 KB, 8 ways, 64 byte lines
		case 0x48:
			l2 = 3072;
			break; // code and data L2 cache, 3072 KB, 12 ways, 64 byte lines
		case 0x49:
			if (l2 != 0)
				l3 = 4096;
			else {
				check_for_p2_core2 = true;
				l3 = l2 = 4096;
			}
			break; // code and data L3 cache, 4096 KB, 16 ways, 64 byte lines (P4) or L2 for core2
		case 0x4A:
			l3 = 6144;
			break; // code and data L3 cache, 6144 KB, 12 ways, 64 byte lines
		case 0x4B:
			l3 = 8192;
			break; // code and data L3 cache, 8192 KB, 16 ways, 64 byte lines
		case 0x4C:
			l3 = 12288;
			break; // code and data L3 cache, 12288 KB, 12 ways, 64 byte lines
		case 0x4D:
			l3 = 16384;
			break; // code and data L3 cache, 16384 KB, 16 ways, 64 byte lines
		case 0x4E:
			l2 = 6144;
			break; // code and data L2 cache, 6144 KB, 24 ways, 64 byte lines
		case 0x78:
			l2 = 1024;
			break; // code and data L2 cache, 1024 KB, 4 ways, 64 byte lines
		case 0x79:
			l2 = 128;
			break; // code and data L2 cache, 128 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x7A:
			l2 = 256;
			break; // code and data L2 cache, 256 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x7B:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x7C:
			l2 = 1024;
			break; // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines, dual-sectored
		case 0x7D:
			l2 = 2048;
			break; // code and data L2 cache, 2048 KB, 8 ways, 64 byte lines
		case 0x7E:
			l2 = 256;
			break; // code and data L2 cache, 256 KB, 8 ways, 128 byte lines, sect. (IA-64)
		case 0x7F:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 2 ways, 64 byte lines
		case 0x80:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 8 ways, 64 byte lines
		case 0x81:
			l2 = 128;
			break; // code and data L2 cache, 128 KB, 8 ways, 32 byte lines
		case 0x82:
			l2 = 256;
			break; // code and data L2 cache, 256 KB, 8 ways, 32 byte lines
		case 0x83:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 8 ways, 32 byte lines
		case 0x84:
			l2 = 1024;
			break; // code and data L2 cache, 1024 KB, 8 ways, 32 byte lines
		case 0x85:
			l2 = 2048;
			break; // code and data L2 cache, 2048 KB, 8 ways, 32 byte lines
		case 0x86:
			l2 = 512;
			break; // code and data L2 cache, 512 KB, 4 ways, 64 byte lines
		case 0x87:
			l2 = 1024;
			break; // code and data L2 cache, 1024 KB, 8 ways, 64 byte lines
		case 0x88:
			l3 = 2048;
			break; // code and data L3 cache, 2048 KB, 4 ways, 64 byte lines (IA-64)
		case 0x89:
			l3 = 4096;
			break; // code and data L3 cache, 4096 KB, 4 ways, 64 byte lines (IA-64)
		case 0x8A:
			l3 = 8192;
			break; // code and data L3 cache, 8192 KB, 4 ways, 64 byte lines (IA-64)
		case 0x8D:
			l3 = 3072;
			break; // code and data L3 cache, 3072 KB, 12 ways, 128 byte lines (IA-64)

		default:
			break;
		}
	}
	if (check_for_p2_core2 && l2 == l3)
		l3 = 0;
	l1 *= 1024;
	l2 *= 1024;
	l3 *= 1024;
}

void eigen_query_cache_amd(int& l1, int& l2, int& l3) {
	int abcd[4];
	abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
	EIGEN_CPUID(abcd, 0x80000005, 0);
	l1 = (abcd[2] >> 24) * 1024; // C[31:24] = L1 size in KB
	abcd[0] = abcd[1] = abcd[2] = abcd[3] = 0;
	EIGEN_CPUID(abcd, 0x80000006, 0);
	l2 = (abcd[2] >> 16) * 1024;                     // C[31;16] = l2 cache size in KB
	l3 = ((abcd[3] & 0xFFFC000) >> 18) * 512 * 1024; // D[31;18] = l3 cache size in 512KB
}
#endif

void eigen_query_cache_sizes(int& l1, int& l2, int& l3) {
#ifdef EIGEN_CPUID
	int abcd[4];
	int const GenuineIntel[] = {0x756e6547, 0x49656e69, 0x6c65746e};
	int const AuthenticAMD[] = {0x68747541, 0x69746e65, 0x444d4163};
	int const AMDisbetter_[] = {0x69444d41, 0x74656273, 0x21726574}; // "AMDisbetter!"
  (void)GenuineIntel;

	// identify the CPU vendor
	EIGEN_CPUID(abcd, 0x0, 0);
	int max_std_funcs = abcd[1];
	if ((eigen_cpuid_is_vendor)(abcd, AuthenticAMD) || (eigen_cpuid_is_vendor)(abcd, AMDisbetter_)) {
		(eigen_query_cache_amd)(l1, l2, l3);
	} else {
		// by default let's use Intel's API
		if (max_std_funcs >= 4) {
			(eigen_query_cache_intel_direct)(l1, l2, l3);
		} else {
			(eigen_query_cache_intel_codes)(l1, l2, l3);
		}
	}

	// here is the list of other vendors:
	//   ||cpuid_is_vendor(abcd,"VIA VIA VIA ")
	//   ||cpuid_is_vendor(abcd,"CyrixInstead")
	//   ||cpuid_is_vendor(abcd,"CentaurHauls")
	//   ||cpuid_is_vendor(abcd,"GenuineTMx86")
	//   ||cpuid_is_vendor(abcd,"TransmetaCPU")
	//   ||cpuid_is_vendor(abcd,"RiseRiseRise")
	//   ||cpuid_is_vendor(abcd,"Geode by NSC")
	//   ||cpuid_is_vendor(abcd,"SiS SiS SiS ")
	//   ||cpuid_is_vendor(abcd,"UMC UMC UMC ")
	//   ||cpuid_is_vendor(abcd,"NexGenDriven")
#else
	l1 = l2 = l3 = -1;
#endif
}

} // namespace internal
auto cache_size_impl() noexcept -> CacheSize {
	int l1 = -1;
	int l2 = -1;
	int l3 = -1;
	internal::eigen_query_cache_sizes(l1, l2, l3);
	return {l1, l2, l3};
}

} // namespace FAER_ABI_VERSION
} // namespace fae

#include "faer/internal/epilogue.hpp"
