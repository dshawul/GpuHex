
#define GPU __CUDACC__

#include <string>
#ifdef GPU
#	include <cuda.h>
#else
#	include <omp.h>
#	include <math.h>
#	include <ctime>
#endif

//
// parameters
//
#ifdef GPU
#	define nThreads   64
#	define nBlocks    84
#	define nLoop      16
#else
#	define nThreads    1
#	define nBlocks     1
#	define nLoop      16
#endif
#define TT_SIZE   4194304
#define UCTK      0.44f
#define FPU       1.10f

//
// printf
//
#ifdef GPU
#	include "cuPrintf.cu"
#	define print(format, ...) cuPrintf(format, __VA_ARGS__)
#else
#	define print(format, ...) printf(format, __VA_ARGS__)
#endif

//
// locks
//
#ifdef GPU
#	define LOCK          int
#	define l_create(x)   ((x) = 0)
#	define l_trylock(x)  (atomicExch(&(x),1))
#	define l_lock(x)     while(l_trylock(x) != 0);
#	define l_unlock(x)   (atomicExch(&(x),0))
#	define l_add(x,v)	 (atomicAdd(&x,v))
#	define l_sub(x,v)	 (atomicSub(&x,v))
#	define l_barrier()   __syncthreads()
#else
#	define LOCK          omp_lock_t
#	define l_create(x)   omp_init_lock(&x)
#	define l_trylock(x)  omp_test_lock(&x)
#	define l_lock(x)     omp_set_lock(&x)
#	define l_unlock(x)   omp_unset_lock(&x)   
template <class T>
inline void l_add(T x,T v) { 
	#pragma omp atomic 
		x+=v;
}
template <class T>
inline void l_sub(T x,T v) { 
	#pragma omp atomic 
		x-=v;
}
inline void l_barrier() { 
	#pragma omp barrier 
}
#endif

//
// undef cuda specific code
//
#ifndef GPU
#	undef  __host__
#	undef  __device__
#   undef  __global__
#   undef  __shared__
#   undef  __constant__
#	define __host__
#	define __device__
#	define __global__
#   define __shared__
#   define __constant__
#if defined (__GNUC__)
#	define __align__(x)  __attribute__ ((aligned(x)))
#else
#	define __align__(x) __declspec(align(x))
#endif
#endif

//
// types
//
#ifdef _MSC_VER
	typedef unsigned __int64 U64;
	typedef unsigned int U32;
#	define U64(x) (x##ui64)
#	define FMTU64 "0x%016I64x"
#else
#   include <inttypes.h>
	typedef uint64_t U64;
	typedef uint32_t U32;
#	define U64(x) (x##ull)
#	define FMTU64 "0x%016llx"
#endif

//
// define board game
//
typedef U64 MOVE;

struct BOARD {
	U64 wpawns;
	U64 all;
	U32 randn;
	char player;
	char emptyc;

	U32 playout(const BOARD&);
	void make_random_move();
	bool is_white_win();

	__device__ __host__
	void clear() {
		wpawns = 0;
		all = U64(0xffffffffffffffff);
		emptyc = 64;
		player = 0;
	}

	__host__ __device__
	void copy(const BOARD& b) {
		wpawns = b.wpawns;
		all = b.all;
		player = b.player;
		emptyc = b.emptyc;
	}

	__device__ __host__
	void do_move(const MOVE& move) {
		all ^= move;
		if(player == 0)
			wpawns ^= move;
		player ^= 1;
		emptyc--;
	}

	__device__ __host__
	void seed(int sd) {
		randn = sd;
	}

	__device__ __host__
	U32 rand() {
		randn *= 214013;
		randn += 2531011;
		return ((randn >> 16) & 0x7fff);
	}

	__device__ __host__
	U64 rand64() {
		return((U64)rand()) ^ 
			  ((U64)rand() << 15) ^ ((U64)rand() << 30) ^
			  ((U64)rand() << 45) ^ ((U64)rand() << 60);
	}
};


__device__ __host__
void BOARD::make_random_move() {
	U32 rbit = rand() % emptyc;
	U64 mbit = all;
	for(U32 i = 0;i < rbit;i++)
		mbit &= mbit - 1; 
	mbit = mbit & -mbit;

	if(player == 0)
		wpawns ^= mbit;
	all ^= mbit;
	player ^= 1;
	emptyc--;
}

__device__ __host__
bool BOARD::is_white_win(){
	U64 m = (wpawns & U64(0x00000000000000ff)),oldm;
	do {
		oldm = m;
		m |=((((m << 8) | (m >> 8)) | 
			 (((m << 9) | (m << 1)) & U64(0xfefefefefefefefe)) | 
			 (((m >> 9) | (m >> 1)) & U64(0x7f7f7f7f7f7f7f7f))) 
			 & wpawns
			);
		if(m & U64(0xff00000000000000)) {
			return true;
		}
	} while(m != oldm);
	return false;
}

__device__ __host__
U32 BOARD::playout(const BOARD& b) {
	U32 wins = 0;
	for(U32 i = 0;i < nLoop;i++) {
		this->copy(b);

		while(emptyc > 0)
			make_random_move();
			
		if(is_white_win())
			wins++;
	}
	return wins;
}

__constant__
unsigned int index64[64];

__device__ __host__
unsigned int firstone(U64 bb) {
	unsigned int folded;
	bb ^= bb - 1;
	folded = (int) bb ^ (bb >> 32);
	return index64[folded * 0x78291ACF >> 26];
}

//
//sq to string and vice versa
//
#define file(x)          ((x) & 7)
#define rank(x)          ((x) >> 3)
#define SQ(x,y)          (((x) << 3) + (y))

__device__ __host__
char* sq_str(const int& sq,char* s) {
	int f = file(sq);
	int r = rank(sq);
	*s++ = 'a' + (f);
	*s++ = '1' + (r);
	*s = 0;
	return s;
}

__host__
const char* str_sq(int& sq,const char* is) {
	const char* s = is;
	int f = tolower(*s++) - 'a';
	int r = atoi(s++) - 1;
	sq = SQ(r,f);
	return s;
}

//
// Node
//

struct Node {
	MOVE move;
	U32 uct_wins;
	U32 uct_visits;
	Node* parent;
	Node* child;
	Node* next;
	LOCK lock;
	int workers;
	
	__device__ __host__
	void clear() {
		uct_wins = 0;
		uct_visits = 0;
		parent = 0;
		child = 0;
		next = 0;
		move = MOVE();
		l_create(lock);
		workers = 0;
	}
};

//
// Table
//

namespace TABLE {
	__device__ Node* mem_;
	__device__ int tsize;
	__device__ BOARD root_board;
	__device__ Node* root_node;
	__device__ Node* head;
	__device__ int size;
	__device__ LOCK lock;
	Node* hmem_;

	__device__ Node* get_node() {
		if(size > 0) {	
			l_lock(lock);
			if(size > 0) {
				size--;
				head++;
				head->clear();
			} else 
				head = 0;
			l_unlock(lock);
			return head;
		} else {
			return 0;
		}
	}
	__global__ void reset() {
		head = mem_;
		size = tsize;
		root_node = get_node();
	}
	
	__global__ void print_tree(int depthLimit) {
		int depth = 0,max_depth = 0,average_depth = 0;
		int leaf_nodes = 0,total_nodes = 0;
		char str[4];
		int sq;
		Node* current = root_node;
		while(current) {
			while(current) {
				while(current) {

					if(current->uct_visits && depth <= depthLimit) {
						for(int i = 0;i < depth;i++)
							print("\t");
						sq = firstone(current->move);
						sq_str(sq,str);
						print("%d.%s %d %d %.6f\n",
							depth,(const char*)str,
							current->uct_wins,current->uct_visits,
							float(current->uct_wins) / current->uct_visits
							);
					}

					total_nodes++;
					if(current->child) {
						depth++;
						current = current->child;
					} else {
						if(depth > max_depth)
							max_depth = depth;
						average_depth += depth;
						leaf_nodes++;
						break;
					}
				}
NEXT:
				if(current->next) {
					current = current->next;
				} else break;
			}
			if(current->parent) {
				depth--;
				current = current->parent;
				goto NEXT;
			} else {
				break;
			}
		}

		print("Total nodes   : %d\n",total_nodes);
		print("Leaf  nodes   : %d\n",leaf_nodes);
		print("Maximum depth : %d\n",max_depth);
		print("Average depth : %.2f\n",average_depth / float(leaf_nodes));
	}
	__device__ void create_children(BOARD* b,Node* n) {
		l_lock(n->lock);
		if(n->child) {
			l_unlock(n->lock);
			return;
		}

		Node* last = n;
		U64 m = b->all;
		U64 lsb;
		while(m) {
			lsb = m & -m;
		
			Node* node = get_node();
			if(!node) break;
			node->move = lsb;
			node->parent = n;
			if(last == n) last->child = node;
			else last->next = node;
			last = node;
			
			m ^= lsb;
		}

		l_unlock(n->lock);
	}

	__device__ Node* UCT_select(Node* n) {
		Node* bnode = 0;
		Node* current = n->child;
		float bvalue = -1.f,value;
		float logn = logf(float(n->uct_visits + 1));
		while(current) {
			if(current->uct_visits > 0) { 
				value = UCTK * sqrtf(logn / (current->uct_visits + 1))
					+ (current->uct_wins + 1) / (current->uct_visits + 1); 
			} else {
				value = FPU;
			}

			value -= (current->workers / 128.f);

			if(value > bvalue) {
				bvalue = value;
				bnode = current;
			}
			current = current->next;
		}
		return bnode;
	}

	__host__ void allocate(int N) {
		static const unsigned int mindex64[64] = {
			63, 30,  3, 32, 59, 14, 11, 33,
			60, 24, 50,  9, 55, 19, 21, 34,
			61, 29,  2, 53, 51, 23, 41, 18,
			56, 28,  1, 43, 46, 27,  0, 35,
			62, 31, 58,  4,  5, 49, 54,  6,
			15, 52, 12, 40,  7, 42, 45, 16,
			25, 57, 48, 13, 10, 39,  8, 44,
			20, 47, 38, 22, 17, 37, 36, 26
		};
#ifdef GPU
		cudaMalloc((void**) &hmem_,N * sizeof(Node));
		cudaMemcpyToSymbol(tsize,&N,sizeof(int));
		cudaMemcpyToSymbol(mem_,&hmem_,sizeof(Node*));
		cudaMemcpyToSymbol(index64,mindex64,sizeof(mindex64));
#else
		hmem_ = (Node*) malloc(N * sizeof(Node));
		tsize = N;
		mem_ = hmem_;
		memcpy(index64,mindex64,sizeof(mindex64));
		l_create(lock);
#endif
	}
	__host__ void release() {
#ifdef GPU
		cudaFree(hmem_);
#else
		free(hmem_);
#endif
	}
}

//
// playout
//
__global__ 
void playout(U32 N) {
	//
	// create blocks
	//
#ifdef GPU
	{
		const int blockId = blockIdx.x;
#else
#pragma omp parallel num_threads(nBlocks)
	{
		const int blockId = omp_get_thread_num();
#endif
		//
		//shared data with in a block
		//
		__shared__ U32 cache[nThreads];
		__shared__ BOARD sb;
		__shared__ Node* n;
		__shared__ bool finished;

		//
		// create threads and allocate a BOARD on register
		//
#ifdef GPU
		{
			const int threadId = threadIdx.x;
#else
#pragma omp parallel num_threads(nThreads)
		{
			const int threadId = omp_get_thread_num();
			print("Block %d : Thread %d of %d\n",
				blockId,threadId,nThreads);
#endif
			BOARD b;
			b.seed(blockId * nBlocks + threadId);
			//
			//loop forever
			//
			while(true) {

				//get node
				if(threadId == 0) {
					finished = false;
					n = TABLE::root_node;
					sb.copy(TABLE::root_board);

					while(n->child) {
						n = TABLE::UCT_select(n);
						sb.do_move(n->move);
					}

					if(n->uct_visits) {
						TABLE::create_children(&sb,n);
						Node* next = TABLE::UCT_select(n);
						if(next) {
							sb.do_move(next->move);
							n = next;
						}
					}

					l_add(n->workers,1);
				}
				l_barrier();
				b.copy(sb);

				//playout the position
				cache[threadId] = b.playout(sb);

				//reduction 
				l_barrier();
				int p = nThreads;
				int i = (p + 1) / 2;
				while (i != 0) {
					if ((threadId < i) && (threadId + i < p))
						cache[threadId] += cache[threadId + i];
					l_barrier();
					p = i;
					if(p > 1) i = (p + 1) / 2;
					else i = p / 2;
				}

				//update result
				if (threadId == 0) {
					l_sub(n->workers,1);

					U32 score;
					if(sb.player != 0) 
						score = cache[0];
					else
						score = nLoop * nThreads - cache[0];
						
					Node* current = n;
					while(current) {
						l_lock(current->lock);
						current->uct_wins += score;
						current->uct_visits += nLoop * nThreads;
						l_unlock(current->lock);
						score = nLoop * nThreads - score;
						current = current->parent;
					}
					if(TABLE::root_node->uct_visits >= N)
						finished = true;
				}
				l_barrier();
				if(finished)
					break;
			}
			//
			// end of work
			//
		}
	}
}
//
// GPU specific code
//

#ifdef GPU

__host__ 
void simulate(BOARD* b,U32 N) {
	cudaMemcpyToSymbol(TABLE::root_board,b,
		sizeof(BOARD),0,cudaMemcpyHostToDevice);

	TABLE::reset <<<1,1>>> ();
	playout <<<nBlocks,nThreads>>> (N); 
	TABLE::print_tree <<<1,1>>> (1);

	cudaPrintfDisplay();
	printf("Errors: %s\n", 
		cudaGetErrorString(cudaPeekAtLastError()));
}
__host__
void init_device() {
	int count;
	cudaDeviceProp prop;
	cudaGetDeviceCount( &count );
	for (int i=0; i< count; i++) {
		cudaGetDeviceProperties( &prop, i );
		printf( " --- General Information for device %d ---\n", i );
		printf( "Name: %s\n", prop.name );
		printf( "Compute capability: %d.%d\n", prop.major, prop.minor );
		printf( "Clock rate: %d\n", prop.clockRate );
		printf( "Device copy overlap: " );
		if (prop.deviceOverlap)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		printf( "Kernel execition timeout : " );
		if (prop.kernelExecTimeoutEnabled)
			printf( "Enabled\n" );
		else
			printf( "Disabled\n" );
		printf( " --- Memory Information for device %d ---\n", i );
		printf( "Total global mem: %ld\n", prop.totalGlobalMem );
		printf( "Total constant Mem: %ld\n", prop.totalConstMem );
		printf( "Max mem pitch: %ld\n", prop.memPitch );
		printf( "Texture Alignment: %ld\n", prop.textureAlignment );
		printf( " --- MP Information for device %d ---\n", i );
		printf( "Multiprocessor count: %d\n",
			prop.multiProcessorCount );
		printf( "Shared mem per mp: %ld\n", prop.sharedMemPerBlock );
		printf( "Registers per mp: %d\n", prop.regsPerBlock );
		printf( "Threads in warp: %d\n", prop.warpSize );
		printf( "Max threads per block: %d\n",
			prop.maxThreadsPerBlock );
		printf( "Max thread dimensions: (%d, %d, %d)\n",
			prop.maxThreadsDim[0], prop.maxThreadsDim[1],
			prop.maxThreadsDim[2] );
		printf( "Max grid dimensions: (%d, %d, %d)\n",
			prop.maxGridSize[0], prop.maxGridSize[1],
			prop.maxGridSize[2] );
		printf( "\n" );
	}

	printf("nBlocks=%d X nThreads=%d\n",nBlocks,nThreads);
	cudaPrintfInit();
	TABLE::allocate(TT_SIZE);
}
__host__ 
void finalize_device() {
	cudaPrintfEnd();
	TABLE::release();
}

#else

//
// Cpu specific code
//

__host__
void simulate(BOARD* bo,U32 N) {
	TABLE::root_board = *bo;
	TABLE::reset();
	playout(N);
	TABLE::print_tree(1);
}
__host__
void init_device() {
	omp_set_nested(1);
	omp_set_dynamic(0);
	TABLE::allocate(TT_SIZE);
}
__host__
void finalize_device() {
	TABLE::release();
}

#endif

//
// Test
//

__host__
void print_bitboard(U64 b) {
	std::string s = "";
	for(int i=7;i>=0;i--) {
		for(int z = 0; z < 7-i;z++)
			s += " ";
		for(int j=0;j<8;j++) {
			U64 m = (((U64)1) << (i * 8 + j));
			if(b & m) s += "1 ";
			else s += "0 ";
		}
		s += "\n";
	}
	printf("%s",s.c_str());
	printf("\n"FMTU64"\n\n",b);
}

static const char *const commands_recognized[] = {
	"d",
	"go",
	"quit",
	"help",
	NULL
};

int main() {
	init_device();

	BOARD b;
	b.clear();

	char str[64];
	while(true) {
		printf("$: ");
		scanf("%s",&str);
		if(!strcmp(str,"d")) {
			print_bitboard(b.wpawns);
			print_bitboard(b.all);
		} else if(!strcmp(str,"help")) {
			size_t index = 0;
			while (commands_recognized[index]) {
				puts(commands_recognized[index]);
				index++;
			}
		} else if(!strcmp(str,"go")) {
			clock_t start,end;
			start = clock();
			simulate(&b,128 * 28 * 128 * 100);
			end = clock();
			printf("time %d\n",end - start);
		} else if(!strcmp(str,"quit")) {
			break;
		} else {
			int move;
			str_sq(move,str);
			b.do_move((U64(1) << move));
		}
	}

	finalize_device();
}

//
// end
//

