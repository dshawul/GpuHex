#include <cuda.h>
#include <string>
#include <ctime>
#include <cstdio>

using namespace std;

//#define GPU

//
// Define board game
//

typedef unsigned __int64 U64;
typedef unsigned int U32;
typedef U64 MOVE;

#define UINT64(x) (x##ui64)

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
		all = UINT64(0xffffffffffffffff);
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
		wpawns ^= move;
		player ^= 1;
		emptyc--;
	}

	__device__ __host__
	void undo_move(const MOVE& move) {
		all ^= move;
		wpawns ^= move;
		player ^= 1;
		emptyc++;
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
bool BOARD::is_white_win() {
	U64 m = (wpawns & UINT64(0x00000000000000ff)),oldm;
	do {
		oldm = m;
		m |=((((m << 8) | (m >> 8)) | 
			 (((m << 9) | (m << 1)) & UINT64(0xfefefefefefefefe)) | 
			 (((m >> 9) | (m >> 1)) & UINT64(0x7f7f7f7f7f7f7f7f))) 
			 & wpawns
			);
		if(m & UINT64(0xff00000000000000)) {
			return true;
		}
	} while(m != oldm);
	return false;
}

#define nLoop     64

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

//
// GPU specific code
//
#ifdef GPU

#include "cuPrintf.cu"

#define nThreads  32
#define nBlocks   112
#define TT_SIZE   4194304
#define UCTK      0.44
#define FPU       10000
//1.10

//
// Lock
//

#define LOCK          int
#define l_create(x)   ((x) = 0)
#define l_trylock(x)  (atomicExch(&(x),1))
#define l_lock(x)     while(l_trylock(x) != 0);
#define l_unlock(x)   (atomicExch(&(x),0))

//
// Node and table
//

struct Node {
	U32 uct_wins;
	U32 uct_visits;
	MOVE move;
	Node* parent;
	Node* child;
	Node* next;
	LOCK lock;
	
	__device__ void clear() {
		uct_wins = 0;
		uct_visits = 0;
		parent = 0;
		child = 0;
		next = 0;
		move = MOVE();
		l_create(lock);
	}
};

namespace TABLE {
	__device__ int size;
	__device__ int tsize;
	__device__ Node* head;
	__device__ Node* mem_;
	__device__ LOCK lock = 0;
	__device__ BOARD root_board;
	__device__ Node* root_node;
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
		int depth = 0;
		Node* current = root_node;
		while(current) {
			while(current) {
				while(current) {
					if(current->uct_visits) {
						for(int i = 0;i < depth;i++)
							cuPrintf("\t");
						cuPrintf("%d. %d %d %.6f\n",
							depth,current->uct_wins,current->uct_visits,
							float(current->uct_wins) / current->uct_visits
							);
					}
					if(current->child && depth < depthLimit) {
						depth++;
						current = current->child;
					} else break;
				}
				if(current->next) {
					current = current->next;
				} else break;
			}
			if(current->parent) {
				depth--;
				current = current->parent->next;
			} else break;
		}

		cuPrintf("Total nodes in tree: %d\n",head - mem_);
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
			if(value > bvalue) {
				bvalue = value;
				bnode = current;
			}
			current = current->next;
		}
		return bnode;
	}

	__host__ void allocate(int N) {
		cudaMalloc((void**) &hmem_,N * sizeof(Node));
		cudaMemcpyToSymbol(tsize,&N,sizeof(int),0,cudaMemcpyHostToDevice);
		cudaMemcpyToSymbol(mem_,&hmem_,sizeof(Node*),0,cudaMemcpyHostToDevice);
	}
	__host__ void release() {
		cudaFree(hmem_);
	}
}

//
// Global code
//

__global__ 
void playout(int N) {

	__shared__ U32 cache[nThreads];
	__shared__ BOARD sb;
	__shared__ Node* n;
	__shared__ bool finished;
	int threadId = threadIdx.x;
	finished = false;

	//local board
	BOARD b;
	b.seed(blockIdx.x * blockDim.x + threadIdx.x + 1);

	//
	//loop forever
	//
	while(true) {

		//get node
		if(threadId == 0) {
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
		}
		b.copy(sb);
		__syncthreads();

		//playout the position
		cache[threadId] = b.playout(sb);

		//reduction : works for power of 2 block size
		__syncthreads();
		int i = blockDim.x / 2;
		while (i != 0) {
			if (threadId < i)
				cache[threadId] += cache[threadId + i];
			__syncthreads();
			i /= 2;
		}

		//update result
		if (threadId == 0) {
			U32 score;
			if(sb.player == 0) 
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
		__syncthreads();
		if(finished)
			break;
	}
}

//
// Host code
//

__host__ 
void simulate(BOARD* b,U32 N) {
	cudaMemcpyToSymbol(TABLE::root_board,b,
		sizeof(BOARD),0,cudaMemcpyHostToDevice);

	TABLE::reset <<<1,1>>> ();
	playout <<<nBlocks,nThreads>>> (N); 
	TABLE::print_tree <<<1,1>>> (12);

    cudaPrintfDisplay();
	printf("Errors: %s\n", 
		cudaGetErrorString(cudaPeekAtLastError()));
}

__host__
void init_device() {

	//inspect device
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

	//init table & cuPrintf
	printf("nBlocks=%d X nThreads=%d\n",nBlocks,nThreads);
	cudaPrintfInit();
    TABLE::allocate(TT_SIZE);
}
__host__ 
void finalize_device() {

	//finalize
	cudaPrintfEnd();
	TABLE::release();
}

#else
__host__
void simulate(BOARD* b,U32 N) {
	BOARD sb;
	sb.copy(*b);
	b->seed(0);
	U32 wins = 0;
	for(U32 i = 0;i < N;i += nLoop)
		wins += b->playout(sb);
	printf("%u %u %.6f\n",wins,N,float(wins)/N);
}
__host__
void init_device() {
}
__host__
void finalize_device() {
}
#endif

__host__
void print_bitboard(U64 b){
	string s = "";
	for(int i=7;i>=0;i--) {
		for(int z = 0; z < i;z++)
			s += "  ";
		for(int j=0;j<8;j++) {
			U64 m = (((U64)1) << (i * 8 + j));
			if(b & m) s += "1 ";
			else s += "0 ";
		}
		s += "\n";
	}
	s += "\n";
	printf("%s",s.c_str());
}

int main() {
	const U32 nSimulations = 
		8 * 14 * 256 * 1600; 

	init_device();

	BOARD b;
	b.clear();
    clock_t start,end;
	start = clock();
	simulate(&b,nSimulations);
	end = clock();
	printf("time %d\n",end - start);

	finalize_device();
}

//
// end
//

