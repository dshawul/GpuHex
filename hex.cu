
#define GPU __CUDACC__

#define CHESS

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
#	define nBlocks    112
#	define WARP       32
#else
#	define nThreads    1
#	define nBlocks     1
#	define WARP        1
#endif
#define nLoop     16
#define nWarps    (nThreads / WARP)
#define TT_SIZE   4194304
#define UCTK      0.44f
#define UCTN      10
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
// Collection of bitboard tricks suitable for GPUs from
// http://chessprogramming.wikispaces.com/Bitboards
// Thanks to Gerd Isinberg and co.
//
__constant__
unsigned int index64[64];
__constant__
U64 king_attacks[64];
__constant__
U64 knight_attacks[64];
__constant__
U64 diagonal_mask_ex[64];
__constant__
U64 antidiag_mask_ex[64];
__constant__
unsigned char first_rank_attacks[64][8];

__device__ __host__
unsigned int firstone(U64 bb) {
	unsigned int folded;
	bb ^= bb - 1;
	folded = (int) bb ^ (bb >> 32);
	return index64[folded * 0x78291ACF >> 26];
}

__device__ __host__
int popcnt(U64 bb) {
	const U32 k1 = 0x55555555;
	const U32 k2 = 0x33333333;
	const U32 k4 = 0x0f0f0f0f;
	const U32 kf = 0x01010101;
	U32 hi = (U32) (bb >> 32);
	U32 lo = (U32) (bb);
	hi =  hi       - ((hi >> 1)  & k1); 
	hi = (hi & k2) + ((hi >> 2)  & k2); 
	hi = (hi       +  (hi >> 4)) & k4 ;
	lo =  lo       - ((lo >> 1)  & k1); 
	lo = (lo & k2) + ((lo >> 2)  & k2); 
	lo = (lo       +  (lo >> 4)) & k4 ;
	return (int) (((hi + lo) * kf) >> 24);
}

__device__ __host__
U64 rankAttacks( U64 occ, int sq ) { 
   union { U64 b; struct { unsigned int l; unsigned int h; }; } b;
   int f = sq & 7; 
   int r = sq & 24;
   if ( sq < 32 ) { 
      b.l = (((unsigned int)occ) >> (r+1)) & 63; 
      b.l = first_rank_attacks[b.l][f] << r; 
      b.h = 0; 
   } else { 
      b.l = 0; 
      b.h = (((unsigned int)(occ>>32)) >> (r+1)) & 63; 
      b.h = first_rank_attacks[b.h][f] << r; 
   } 
   return b.b; 
} 

__device__ __host__
U64 fileAttacks( U64 occ, int sq ) { 
   union { U64 b; struct { unsigned int l; unsigned int h; }; } b;
   int f = sq & 7; 
   b.b = occ; 
   b.l = (b.l >> f) & 0x01010101; 
   b.h = (b.h >> f) & 0x01010101; 
   b.l = (b.l << 4) + b.h; 
   b.h = (b.l       * 0x10080402) >> 26; 
   b.l = 0x08040201 * first_rank_attacks[b.h][(sq^56)>>3]; 
   b.h =((b.l << 4) & 0x80808080) >> (f^7); 
   b.l = (b.l       & 0x80808080) >> (f^7); 
   return b.b; 
} 

__device__ __host__
U64 diagonalAttacks( U64 occ, int sq ) { 
   union { U64 b; struct { unsigned int l; unsigned int h; }; } b;
   b.b = occ & diagonal_mask_ex[sq]; 
   b.h = ((b.l + b.h) * 0x02020202) >> 26; 
   b.h = b.l = 0x01010101 * first_rank_attacks[b.h][sq&7]; 
   return b.b & diagonal_mask_ex[sq]; 
} 

__device__ __host__
U64 antidiagAttacks( U64 occ, int sq ) { 
   union { U64 b; struct { unsigned int l; unsigned int h; }; } b;
   b.b = occ & antidiag_mask_ex[sq]; 
   b.h = ((b.l + b.h) * 0x02020202) >> 26; 
   b.h = b.l = 0x01010101 * first_rank_attacks[b.h][sq&7]; 
   return b.b & antidiag_mask_ex[sq]; 
} 

#define kingAttacks(sq) king_attacks[sq]
#define knightAttacks(sq) knight_attacks[sq]
#define rookAttacks(occ,sq) (fileAttacks(occ,sq) | rankAttacks(occ,sq))
#define bishopAttacks(occ,sq) (diagonalAttacks(occ,sq) | antidiagAttacks(occ,sq))
#define queenAttacks(occ,sq) (rookAttacks(occ,sq) | bishopAttacks(occ,sq))

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
void str_sq(int& sq,const char* is) {
	const char* s = is;
	int f = tolower(*s++) - 'a';
	int r = atoi(s++) - 1;
	sq = SQ(r,f);
}

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

//
// define board game
//

#ifdef CHESS

static const char piece_name[] = "KQRBNPkqrbnp";
static const char rank_name[] = "12345678";
static const char file_name[] = "abcdefgh";
static const char col_name[] = "WwBb";
static const char cas_name[] = "KQkq";
static const char start_fen[] = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

enum PIECES {
	white,black,king,queen,rook,bishop,knight,pawn
};
enum RANKS {
	RANK1,RANK2,RANK3,RANK4,RANK5,RANK6,RANK7,RANK8
};
enum FILES {
	FILEA,FILEB,FILEC,FILED,FILEE,FILEF,FILEG,FILEH
};
enum CASTLE_STATUS {
	WSC_FLAG = 1, WLC_FLAG = 2, BSC_FLAG = 4, BLC_FLAG = 8,
	WSLC_FLAG = 3, BSLC_FLAG = 12, WBC_FLAG = 15
};

#define m_from(x)  ((x) & 0xff)
#define m_to(x)    (((x) >> 8) & 0xff)
#define m_piece(x) (((x) >> 16) & 0xff)

typedef U32 MOVE;

struct BOARD {
	U64 pieces[8];
	U32 randn;
	unsigned char player;
	unsigned char castle;
	unsigned char epsquare;
	unsigned char fifty;

	__host__
	void clear() {
		set_fen(start_fen);
	}

	__host__ __device__
	void copy(const BOARD& b) {
		memcpy(pieces,b.pieces,sizeof(pieces));
		player = b.player;
		castle = b.castle;
		epsquare = b.epsquare;
		fifty = b.fifty;
	}

	__device__ __host__
	void xor(int sq,int ind) {
		pieces[ind] ^= (U64(1) << sq);
	}

	__device__ __host__
	void xor(int sq,int pic,int col) {
		pieces[col] ^= (U64(1) << sq);
		pieces[pic] ^= (U64(1) << sq);
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

	U32 playout(const BOARD&);
	void make_random_move();
	bool is_white_win();

	void do_move(const MOVE& move);
	int count_moves();
	MOVE gen_move(int);
	bool attacked(U64,int,int);

	void set_fen(const char* fen);
	void get_fen(char* fen) const;
	void str_mov(MOVE& move,const char* is);

	__host__
	void print_board() const{
		char fen[256];
		get_fen(fen);
		printf("%s\n",fen);
	}
};

__device__ __host__
void BOARD::do_move(const MOVE& move) {
	int from = m_from(move);
	int to = m_to(move);
	int pic = m_piece(move);

	xor(from,pic,player);

	U64 toBB = ((U64)1 << to) & pieces[player ^ 1];
	if(toBB) {
		int cap = 0;
		for(int i = pawn;i >= queen;i--) {
			if(toBB & pieces[i]) {
				cap = i;
				break;
			}
		}
		xor(to,cap,player ^ 1);
	}

	xor(to,pic,player);

	player ^= 1;
}

__device__ __host__
int BOARD::count_moves() {
	const U64 occ = (pieces[white] | pieces[black]);
	U64 pieceb,movesb;
	int from,count = 0;

	//pawns
	pieceb = pieces[player] & pieces[pawn];
	if(player == white) {
		movesb = (pieceb << 8) & ~occ;
		movesb |= ((movesb << 8) & ~occ) & U64(0x00000000ff000000);
		count += popcnt(movesb);

		movesb = ((pieceb & U64(0xf7f7f7f7f7f7f7f7)) << 7) & pieces[black];
		count += popcnt(movesb);

		movesb = ((pieceb & U64(0x7f7f7f7f7f7f7f7f)) << 9) & pieces[black];
		count += popcnt(movesb);
	} else {
		movesb = (pieceb >> 8) & ~occ;
		movesb |= ((movesb >> 8) & ~occ) & U64(0x000000ff00000000);
		count += popcnt(movesb);

		movesb = ((pieceb & U64(0x7f7f7f7f7f7f7f7f)) >> 7) & pieces[white];
		count += popcnt(movesb);

		movesb = ((pieceb & U64(0xf7f7f7f7f7f7f7f7)) >> 9) & pieces[white];
		count += popcnt(movesb);
	}

	//legal board
	const U64 legalB = (~occ | pieces[player ^ 1]);

	//knight
	pieceb = pieces[player] & pieces[knight];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = knightAttacks(from) & legalB;
		count += popcnt(movesb);
		pieceb &= pieceb - 1;
	}

	//bishop
	pieceb = pieces[player] & pieces[bishop];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = bishopAttacks(occ,from) & legalB;
		count += popcnt(movesb);
		pieceb &= pieceb - 1;
	}

	//rook
	pieceb = pieces[player] & pieces[rook];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = rookAttacks(occ,from) & legalB;
		count += popcnt(movesb);
		pieceb &= pieceb - 1;
	}

	//queen
	pieceb = pieces[player] & pieces[queen];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = queenAttacks(occ,from) & legalB;
		count += popcnt(movesb);
		pieceb &= pieceb - 1;
	}

	return count;
} 

__device__ __host__
MOVE BOARD::gen_move(int index) {
	const U64 occ = (pieces[white] | pieces[black]);
	U64 pieceb,movesb;
	int from,to,count = 0;

	//move
#define CHECK(pic) {										\
	count++;												\
	if(count == index + 1) {								\
		MOVE move = from | (to << 8) | (pic << 16);			\
		return move;										\
	}														\
};
	//pawns
	pieceb = pieces[player] & pieces[pawn];
	if(player == white) {
		U64 tempb;

		movesb = (pieceb << 8) & ~occ;
		tempb = movesb;
		while(movesb) {
			to = firstone(movesb);
			from = to - 8;
			CHECK(pawn);
			movesb &= movesb - 1;
		}
		movesb = ((tempb << 8) & ~occ) & U64(0x00000000ff000000);
		while(movesb) {
			to = firstone(movesb);
			from = to - 16;
			CHECK(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & U64(0xf7f7f7f7f7f7f7f7)) << 7) & pieces[black];
		while(movesb) {
			to = firstone(movesb);
			from = to - 7;
			CHECK(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & U64(0x7f7f7f7f7f7f7f7f)) << 9) & pieces[black];
		while(movesb) {
			to = firstone(movesb);
			from = to - 9;
			CHECK(pawn);
			movesb &= movesb - 1;
		}
	} else {
		U64 tempb;

		movesb = (pieceb >> 8) & ~occ;
		tempb = movesb;
		while(movesb) {
			to= firstone(movesb);
			from = to + 8;
			CHECK(pawn);
			movesb &= movesb - 1;
		}
		movesb = ((tempb >> 8) & ~occ) & U64(0x000000ff00000000);
		while(movesb) {
			to = firstone(movesb);
			from = to + 16;
			CHECK(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & U64(0x7f7f7f7f7f7f7f7f)) >> 7) & pieces[white];
		while(movesb) {
			to = firstone(movesb);
			from = to + 7;
			CHECK(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & U64(0xf7f7f7f7f7f7f7f7)) >> 9) & pieces[white];
		while(movesb) {
			to = firstone(movesb);
			from = to + 9;
			CHECK(pawn);
			movesb &= movesb - 1;
		}
	}

	//legal board
	const U64 legalB = (~occ | pieces[player ^ 1]);

	//knight
	pieceb = pieces[player] & pieces[knight];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = knightAttacks(from) & legalB;

		while(movesb) {
			to = firstone(movesb);
			CHECK(knight);
			movesb &= movesb - 1;
		}

		pieceb &= pieceb - 1;
	}

	//bishop
	pieceb = pieces[player] & pieces[bishop];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = bishopAttacks(occ,from) & legalB;

		while(movesb) {
			to = firstone(movesb);
			CHECK(bishop);
			movesb &= movesb - 1;
		}

		pieceb &= pieceb - 1;
	}

	//rook
	pieceb = pieces[player] & pieces[rook];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = rookAttacks(occ,from) & legalB;

		while(movesb) {
			to = firstone(movesb);
			CHECK(rook);
			movesb &= movesb - 1;
		}

		pieceb &= pieceb - 1;
	}

	//queen
	pieceb = pieces[player] & pieces[queen];
	while(pieceb) {
		from = firstone(pieceb);
		movesb = queenAttacks(occ,from) & legalB;

		while(movesb) {
			to = firstone(movesb);
			CHECK(queen);
			movesb &= movesb - 1;
		}

		pieceb &= pieceb - 1;
	}

	return MOVE();
}

__device__ __host__
bool BOARD::attacked(U64 occ,int sq,int col) {
	U64 pawns = pieces[col] & pieces[pawn];
	if(col == white) {
		if(file(sq) > FILEA && (((U64)1 << (sq - 9)) & pawns))
			return true;
		if(file(sq) < FILEH && (((U64)1 << (sq - 7)) & pawns))
			return true;
	} else {
		if(file(sq) > FILEA && (((U64)1 << (sq + 7)) & pawns))
			return true;
		if(file(sq) < FILEH && (((U64)1 << (sq + 9)) & pawns))
			return true;
	}
	if(knightAttacks(sq) & pieces[col] & pieces[knight]) return true;
	if(kingAttacks(sq) & pieces[col] & pieces[king]) return true;
	U64 bishopsQueens = pieces[col] & (pieces[bishop] | pieces[queen] );
	if(bishopAttacks(occ, sq) & bishopsQueens) return true;
	U64 rooksQueens = pieces[col] & (pieces[rook] | pieces[queen]);
	if(rookAttacks (occ, sq) & rooksQueens) return true;
	return false;
}

__device__ __host__
void BOARD::make_random_move() {
	U32 N = count_moves();
	U32 index = U32((N) * (float(rand()) / 0x7fff));
	MOVE move = gen_move(index);
	do_move(move);
}

__device__ __host__
bool BOARD::is_white_win(){
	return (rand() & 1);
}

__device__ __host__
U32 BOARD::playout(const BOARD& b) {
	U32 wins = 0;
	for(U32 i = 0;i < nLoop;i++) {
		this->copy(b);

		for(int j = 0;j < 20;j++)
			make_random_move();
			
		if(is_white_win())
			wins++;
	}
	return wins;
}

__host__
void BOARD::set_fen(const char* fen_str) {

	//reset
	memset(this,0,sizeof(BOARD));

	//pieces
	int i,r,f,sq,pic,col,move_number;
	const char* p = fen_str,*pfen;
	for(r = RANK8;r >= RANK1; r--) {
		for(f = FILEA;f <= FILEH;f++) {
			sq = SQ(r,f);
			if((pfen = strchr(piece_name,*p)) != 0) {
				pic = int(strchr(piece_name,*pfen) - piece_name);
				col = (pic >= 6);
				pic = (pic % 6) + 2;
				xor(sq,pic,col);
			} else if((pfen = strchr(rank_name,*p)) != 0) {
				for(i = 0;i < pfen - rank_name;i++) {
					f++;
				}
			} 
			p++;
		}
		p++;
	}

	//player
	if((pfen = strchr(col_name,*p)) != 0)
		player = ((pfen - col_name) >= 2);
	p++;
	p++;

	//castling rights
	castle = 0;
	if(*p == '-') {
		p++;
	} else {
		while((pfen = strchr(cas_name,*p)) != 0) {
			castle |= (1 << (pfen - cas_name));
			p++;
		}
	}

	//epsquare
	p++;
	if(*p == '-') {
		epsquare = 0;
		p++;
	} else {
		epsquare = int(strchr(file_name,*p) - file_name);
		p++;
		epsquare += 16 * int(strchr(rank_name,*p) - rank_name);
		p++;
	}

	//fifty & hply
	p++;
	if(*p) sscanf(p,"%d %d",&fifty,&move_number);
	else {
		fifty = 0;
		move_number = 1;
	}	
}

__host__
void BOARD::get_fen(char* fen) const {
	int f,r,sq,pic;

	//pieces
	for(r = 7; r >= 0 ; --r) {
		for (f = 0; f <= 7 ; f++) {
			sq = SQ(r,f);
			pic = 0;
			for(int k = king;k <= pawn;k++) {
				if(pieces[k] & ((U64)1 << sq)) {
					pic = k;
					break;
				}
			}
			if(pic) {
				pic -= 2;
				if(pieces[black] & ((U64)1 << sq))
					pic += 6;
				*fen++ = piece_name[pic];
			} else {
				*fen++ = '-';
			}
		}
		if(r!=0) *fen++ = '\n';
	}
	*fen++ = '\n';

	//player
	*fen++ = ' ';
	if(player == white) *fen++ = 'w';
	else *fen++ = 'b';
	*fen++ = ' ';

	//castling
	if(!castle) *fen++ = '-';
	else {
		if(castle & WSC_FLAG) *fen++ = 'K';
		if(castle & WLC_FLAG) *fen++ = 'Q';
		if(castle & BSC_FLAG) *fen++ = 'k';
		if(castle & BLC_FLAG) *fen++ = 'q';
	}
	*fen++ = ' ';

	//enpassant
	if (!epsquare) *fen++ = '-';
	else {
		*fen++ = file_name[file(epsquare)];
		*fen++ = rank_name[rank(epsquare)];
	}

	*fen = 0;

	//fifty & hply
	char str[12];
	int move_number = 1;
	sprintf(str," %d %d",fifty,move_number);
	strcat(fen,str);
}

__device__ __host__
char* mov_str(const MOVE& move,char* s) {
	s = sq_str(m_from(move),s);
	s = sq_str(m_to(move),s);
	return s;
}
__host__
void BOARD::str_mov(MOVE& move,const char* is) {
	int from,to,pic = 0;
	str_sq(from,is);
	str_sq(to,&is[2]);

	U64 fromBB = ((U64)1 << from) & pieces[player];
	if(fromBB) {
		for(int i = pawn;i >= queen;i--) {
			if(fromBB & pieces[i]) {
				pic = i;
				break;
			}
		}
	}
	move = from | (to << 8) | (pic << 16);
}
#else

typedef U64 MOVE;

struct BOARD {
	U64 wpawns;
	U64 all;
	U32 randn;
	char player;
	char emptyc;

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

	__host__
	void print_board() const {
		print_bitboard(wpawns);
		print_bitboard(all);
	}

	U32 playout(const BOARD&);
	void make_random_move();
	bool is_white_win();

	int count_moves();
	MOVE gen_move(int);

	void str_mov(MOVE& move,const char* is);
};


__device__ __host__
void BOARD::make_random_move() {
	U32 rbit = rand() % emptyc;
	U64 move = all;
	for(U32 i = 0;i < rbit;i++)
		move &= move - 1; 
	move = move & -move;

	do_move(move);
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

__device__ __host__
int BOARD::count_moves() {
	return emptyc;
}

__device__ __host__
MOVE BOARD::gen_move(int index) {
	int count = 0;
	U64 m = all;
	while(m) {
		if(count == index)
			return MOVE(m & -m);
		count++;
		m &= m - 1;
	}
	return MOVE();
}

__device__ __host__
char* mov_str(const MOVE& move,char* s) {
	int sq = firstone(move);
	s = sq_str(sq,s);
	return s;
}
__host__
void BOARD::str_mov(MOVE& move,const char* is) {
	int sq;
	str_sq(sq,is);
	move = ((U64)1 << sq);
}
#endif

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
		char str[8];
		Node* current = root_node;
		while(current) {
			while(current) {
				while(current) {

					if(current->uct_visits && depth <= depthLimit) {
						for(int i = 0;i < depth;i++)
							print("\t");
						mov_str(current->move,str);
						print("%d.%s %12d %12d %12.6f\n",
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

		Node *last = n,*node;
		MOVE move;
		int N = b->count_moves();
		for(int i = 0;i < N;i++) {
			move = b->gen_move(i);
			node = get_node();
			if(!node) break;
			node->move = move;
			node->parent = n;
			if(last == n) last->child = node;
			else last->next = node;
			last = node;
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
					+ (current->uct_wins + 1) / float(current->uct_visits + 1); 
			} else {
				value = FPU;
			}

			value -= (current->workers / 1024.f);

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
		static const U64 mknight_attacks[64] = {
			U64(0x0000000000020400),U64(0x0000000000050800),U64(0x00000000000a1100),U64(0x0000000000142200),
			U64(0x0000000000284400),U64(0x0000000000508800),U64(0x0000000000a01000),U64(0x0000000000402000),
			U64(0x0000000002040004),U64(0x0000000005080008),U64(0x000000000a110011),U64(0x0000000014220022),
			U64(0x0000000028440044),U64(0x0000000050880088),U64(0x00000000a0100010),U64(0x0000000040200020),
			U64(0x0000000204000402),U64(0x0000000508000805),U64(0x0000000a1100110a),U64(0x0000001422002214),
			U64(0x0000002844004428),U64(0x0000005088008850),U64(0x000000a0100010a0),U64(0x0000004020002040),
			U64(0x0000020400040200),U64(0x0000050800080500),U64(0x00000a1100110a00),U64(0x0000142200221400),
			U64(0x0000284400442800),U64(0x0000508800885000),U64(0x0000a0100010a000),U64(0x0000402000204000),
			U64(0x0002040004020000),U64(0x0005080008050000),U64(0x000a1100110a0000),U64(0x0014220022140000),
			U64(0x0028440044280000),U64(0x0050880088500000),U64(0x00a0100010a00000),U64(0x0040200020400000),
			U64(0x0204000402000000),U64(0x0508000805000000),U64(0x0a1100110a000000),U64(0x1422002214000000),
			U64(0x2844004428000000),U64(0x5088008850000000),U64(0xa0100010a0000000),U64(0x4020002040000000),
			U64(0x0400040200000000),U64(0x0800080500000000),U64(0x1100110a00000000),U64(0x2200221400000000),
			U64(0x4400442800000000),U64(0x8800885000000000),U64(0x100010a000000000),U64(0x2000204000000000),
			U64(0x0004020000000000),U64(0x0008050000000000),U64(0x00110a0000000000),U64(0x0022140000000000),
			U64(0x0044280000000000),U64(0x0088500000000000),U64(0x0010a00000000000),U64(0x0020400000000000)
		};
		static const U64 mking_attacks[64] = {
			U64(0x0000000000000302),U64(0x0000000000000705),U64(0x0000000000000e0a),U64(0x0000000000001c14),
			U64(0x0000000000003828),U64(0x0000000000007050),U64(0x000000000000e0a0),U64(0x000000000000c040),
			U64(0x0000000000030203),U64(0x0000000000070507),U64(0x00000000000e0a0e),U64(0x00000000001c141c),
			U64(0x0000000000382838),U64(0x0000000000705070),U64(0x0000000000e0a0e0),U64(0x0000000000c040c0),
			U64(0x0000000003020300),U64(0x0000000007050700),U64(0x000000000e0a0e00),U64(0x000000001c141c00),
			U64(0x0000000038283800),U64(0x0000000070507000),U64(0x00000000e0a0e000),U64(0x00000000c040c000),
			U64(0x0000000302030000),U64(0x0000000705070000),U64(0x0000000e0a0e0000),U64(0x0000001c141c0000),
			U64(0x0000003828380000),U64(0x0000007050700000),U64(0x000000e0a0e00000),U64(0x000000c040c00000),
			U64(0x0000030203000000),U64(0x0000070507000000),U64(0x00000e0a0e000000),U64(0x00001c141c000000),
			U64(0x0000382838000000),U64(0x0000705070000000),U64(0x0000e0a0e0000000),U64(0x0000c040c0000000),
			U64(0x0003020300000000),U64(0x0007050700000000),U64(0x000e0a0e00000000),U64(0x001c141c00000000),
			U64(0x0038283800000000),U64(0x0070507000000000),U64(0x00e0a0e000000000),U64(0x00c040c000000000),
			U64(0x0302030000000000),U64(0x0705070000000000),U64(0x0e0a0e0000000000),U64(0x1c141c0000000000),
			U64(0x3828380000000000),U64(0x7050700000000000),U64(0xe0a0e00000000000),U64(0xc040c00000000000),
			U64(0x0203000000000000),U64(0x0507000000000000),U64(0x0a0e000000000000),U64(0x141c000000000000),
			U64(0x2838000000000000),U64(0x5070000000000000),U64(0xa0e0000000000000),U64(0x40c0000000000000)
		};
		static const U64 mdiagonal_mask_ex[64] = {
			U64(0x8040201008040200),U64(0x0080402010080400),U64(0x0000804020100800),U64(0x0000008040201000),
			U64(0x0000000080402000),U64(0x0000000000804000),U64(0x0000000000008000),U64(0x0000000000000000),
			U64(0x4020100804020000),U64(0x8040201008040001),U64(0x0080402010080002),U64(0x0000804020100004),
			U64(0x0000008040200008),U64(0x0000000080400010),U64(0x0000000000800020),U64(0x0000000000000040),
			U64(0x2010080402000000),U64(0x4020100804000100),U64(0x8040201008000201),U64(0x0080402010000402),
			U64(0x0000804020000804),U64(0x0000008040001008),U64(0x0000000080002010),U64(0x0000000000004020),
			U64(0x1008040200000000),U64(0x2010080400010000),U64(0x4020100800020100),U64(0x8040201000040201),
			U64(0x0080402000080402),U64(0x0000804000100804),U64(0x0000008000201008),U64(0x0000000000402010),
			U64(0x0804020000000000),U64(0x1008040001000000),U64(0x2010080002010000),U64(0x4020100004020100),
			U64(0x8040200008040201),U64(0x0080400010080402),U64(0x0000800020100804),U64(0x0000000040201008),
			U64(0x0402000000000000),U64(0x0804000100000000),U64(0x1008000201000000),U64(0x2010000402010000),
			U64(0x4020000804020100),U64(0x8040001008040201),U64(0x0080002010080402),U64(0x0000004020100804),
			U64(0x0200000000000000),U64(0x0400010000000000),U64(0x0800020100000000),U64(0x1000040201000000),
			U64(0x2000080402010000),U64(0x4000100804020100),U64(0x8000201008040201),U64(0x0000402010080402),
			U64(0x0000000000000000),U64(0x0001000000000000),U64(0x0002010000000000),U64(0x0004020100000000),
			U64(0x0008040201000000),U64(0x0010080402010000),U64(0x0020100804020100),U64(0x0040201008040201)
		};
		static const U64 mantidiag_mask_ex[64] = {
			U64(0x0000000000000000),U64(0x0000000000000100),U64(0x0000000000010200),U64(0x0000000001020400),
			U64(0x0000000102040800),U64(0x0000010204081000),U64(0x0001020408102000),U64(0x0102040810204000),
			U64(0x0000000000000002),U64(0x0000000000010004),U64(0x0000000001020008),U64(0x0000000102040010),
			U64(0x0000010204080020),U64(0x0001020408100040),U64(0x0102040810200080),U64(0x0204081020400000),
			U64(0x0000000000000204),U64(0x0000000001000408),U64(0x0000000102000810),U64(0x0000010204001020),
			U64(0x0001020408002040),U64(0x0102040810004080),U64(0x0204081020008000),U64(0x0408102040000000),
			U64(0x0000000000020408),U64(0x0000000100040810),U64(0x0000010200081020),U64(0x0001020400102040),
			U64(0x0102040800204080),U64(0x0204081000408000),U64(0x0408102000800000),U64(0x0810204000000000),
			U64(0x0000000002040810),U64(0x0000010004081020),U64(0x0001020008102040),U64(0x0102040010204080),
			U64(0x0204080020408000),U64(0x0408100040800000),U64(0x0810200080000000),U64(0x1020400000000000),
			U64(0x0000000204081020),U64(0x0001000408102040),U64(0x0102000810204080),U64(0x0204001020408000),
			U64(0x0408002040800000),U64(0x0810004080000000),U64(0x1020008000000000),U64(0x2040000000000000),
			U64(0x0000020408102040),U64(0x0100040810204080),U64(0x0200081020408000),U64(0x0400102040800000),
			U64(0x0800204080000000),U64(0x1000408000000000),U64(0x2000800000000000),U64(0x4000000000000000),
			U64(0x0002040810204080),U64(0x0004081020408000),U64(0x0008102040800000),U64(0x0010204080000000),
			U64(0x0020408000000000),U64(0x0040800000000000),U64(0x0080000000000000),U64(0x0000000000000000)
		};
		const unsigned char mfirst_rank_attacks[64][8] = {
			0xfe,0xfd,0xfb,0xf7,0xef,0xdf,0xbf,0x7f,0x02,0xfd,0xfa,0xf6,0xee,0xde,0xbe,0x7e,
			0x06,0x05,0xfb,0xf4,0xec,0xdc,0xbc,0x7c,0x02,0x05,0xfa,0xf4,0xec,0xdc,0xbc,0x7c,
			0x0e,0x0d,0x0b,0xf7,0xe8,0xd8,0xb8,0x78,0x02,0x0d,0x0a,0xf6,0xe8,0xd8,0xb8,0x78,
			0x06,0x05,0x0b,0xf4,0xe8,0xd8,0xb8,0x78,0x02,0x05,0x0a,0xf4,0xe8,0xd8,0xb8,0x78,
			0x1e,0x1d,0x1b,0x17,0xef,0xd0,0xb0,0x70,0x02,0x1d,0x1a,0x16,0xee,0xd0,0xb0,0x70,
			0x06,0x05,0x1b,0x14,0xec,0xd0,0xb0,0x70,0x02,0x05,0x1a,0x14,0xec,0xd0,0xb0,0x70,
			0x0e,0x0d,0x0b,0x17,0xe8,0xd0,0xb0,0x70,0x02,0x0d,0x0a,0x16,0xe8,0xd0,0xb0,0x70,
			0x06,0x05,0x0b,0x14,0xe8,0xd0,0xb0,0x70,0x02,0x05,0x0a,0x14,0xe8,0xd0,0xb0,0x70,
			0x3e,0x3d,0x3b,0x37,0x2f,0xdf,0xa0,0x60,0x02,0x3d,0x3a,0x36,0x2e,0xde,0xa0,0x60,
			0x06,0x05,0x3b,0x34,0x2c,0xdc,0xa0,0x60,0x02,0x05,0x3a,0x34,0x2c,0xdc,0xa0,0x60,
			0x0e,0x0d,0x0b,0x37,0x28,0xd8,0xa0,0x60,0x02,0x0d,0x0a,0x36,0x28,0xd8,0xa0,0x60,
			0x06,0x05,0x0b,0x34,0x28,0xd8,0xa0,0x60,0x02,0x05,0x0a,0x34,0x28,0xd8,0xa0,0x60,
			0x1e,0x1d,0x1b,0x17,0x2f,0xd0,0xa0,0x60,0x02,0x1d,0x1a,0x16,0x2e,0xd0,0xa0,0x60,
			0x06,0x05,0x1b,0x14,0x2c,0xd0,0xa0,0x60,0x02,0x05,0x1a,0x14,0x2c,0xd0,0xa0,0x60,
			0x0e,0x0d,0x0b,0x17,0x28,0xd0,0xa0,0x60,0x02,0x0d,0x0a,0x16,0x28,0xd0,0xa0,0x60,
			0x06,0x05,0x0b,0x14,0x28,0xd0,0xa0,0x60,0x02,0x05,0x0a,0x14,0x28,0xd0,0xa0,0x60,
			0x7e,0x7d,0x7b,0x77,0x6f,0x5f,0xbf,0x40,0x02,0x7d,0x7a,0x76,0x6e,0x5e,0xbe,0x40,
			0x06,0x05,0x7b,0x74,0x6c,0x5c,0xbc,0x40,0x02,0x05,0x7a,0x74,0x6c,0x5c,0xbc,0x40,
			0x0e,0x0d,0x0b,0x77,0x68,0x58,0xb8,0x40,0x02,0x0d,0x0a,0x76,0x68,0x58,0xb8,0x40,
			0x06,0x05,0x0b,0x74,0x68,0x58,0xb8,0x40,0x02,0x05,0x0a,0x74,0x68,0x58,0xb8,0x40,
			0x1e,0x1d,0x1b,0x17,0x6f,0x50,0xb0,0x40,0x02,0x1d,0x1a,0x16,0x6e,0x50,0xb0,0x40,
			0x06,0x05,0x1b,0x14,0x6c,0x50,0xb0,0x40,0x02,0x05,0x1a,0x14,0x6c,0x50,0xb0,0x40,
			0x0e,0x0d,0x0b,0x17,0x68,0x50,0xb0,0x40,0x02,0x0d,0x0a,0x16,0x68,0x50,0xb0,0x40,
			0x06,0x05,0x0b,0x14,0x68,0x50,0xb0,0x40,0x02,0x05,0x0a,0x14,0x68,0x50,0xb0,0x40,
			0x3e,0x3d,0x3b,0x37,0x2f,0x5f,0xa0,0x40,0x02,0x3d,0x3a,0x36,0x2e,0x5e,0xa0,0x40,
			0x06,0x05,0x3b,0x34,0x2c,0x5c,0xa0,0x40,0x02,0x05,0x3a,0x34,0x2c,0x5c,0xa0,0x40,
			0x0e,0x0d,0x0b,0x37,0x28,0x58,0xa0,0x40,0x02,0x0d,0x0a,0x36,0x28,0x58,0xa0,0x40,
			0x06,0x05,0x0b,0x34,0x28,0x58,0xa0,0x40,0x02,0x05,0x0a,0x34,0x28,0x58,0xa0,0x40,
			0x1e,0x1d,0x1b,0x17,0x2f,0x50,0xa0,0x40,0x02,0x1d,0x1a,0x16,0x2e,0x50,0xa0,0x40,
			0x06,0x05,0x1b,0x14,0x2c,0x50,0xa0,0x40,0x02,0x05,0x1a,0x14,0x2c,0x50,0xa0,0x40,
			0x0e,0x0d,0x0b,0x17,0x28,0x50,0xa0,0x40,0x02,0x0d,0x0a,0x16,0x28,0x50,0xa0,0x40,
			0x06,0x05,0x0b,0x14,0x28,0x50,0xa0,0x40,0x02,0x05,0x0a,0x14,0x28,0x50,0xa0,0x40
		};
#ifdef GPU
		cudaMalloc((void**) &hmem_,N * sizeof(Node));
		cudaMemcpyToSymbol(tsize,&N,sizeof(int));
		cudaMemcpyToSymbol(mem_,&hmem_,sizeof(Node*));
		cudaMemcpyToSymbol(index64,mindex64,sizeof(mindex64));
		cudaMemcpyToSymbol(knight_attacks,mknight_attacks,sizeof(mknight_attacks));
		cudaMemcpyToSymbol(king_attacks,mking_attacks,sizeof(mking_attacks));
		cudaMemcpyToSymbol(diagonal_mask_ex,mdiagonal_mask_ex,sizeof(mdiagonal_mask_ex));
		cudaMemcpyToSymbol(antidiag_mask_ex,mantidiag_mask_ex,sizeof(mantidiag_mask_ex));
		cudaMemcpyToSymbol(first_rank_attacks,mfirst_rank_attacks,sizeof(mfirst_rank_attacks));
#else
		hmem_ = (Node*) malloc(N * sizeof(Node));
		tsize = N;
		mem_ = hmem_;
		memcpy(index64,mindex64,sizeof(mindex64));
		memcpy(knight_attacks,mknight_attacks,sizeof(mknight_attacks));
		memcpy(king_attacks,mking_attacks,sizeof(mking_attacks));
		memcpy(diagonal_mask_ex,mdiagonal_mask_ex,sizeof(mdiagonal_mask_ex));
		memcpy(antidiag_mask_ex,mantidiag_mask_ex,sizeof(mantidiag_mask_ex));
		memcpy(first_rank_attacks,mfirst_rank_attacks,sizeof(mfirst_rank_attacks));
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
		__shared__ BOARD sbw[nWarps];
		__shared__ Node* nw[nWarps];
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
			BOARD& sb = sbw[threadId / WARP];
			Node*& n = nw[threadId / WARP];
			b.seed(blockId * nBlocks + threadId);
			finished = false;
			const int threadIdWarp = threadId & (WARP - 1);

			//
			//loop forever
			//
			while(true) {

				//get node
				if(threadIdWarp == 0) {
					n = TABLE::root_node;
					sb.copy(TABLE::root_board);

					while(n->child) {
						n = TABLE::UCT_select(n);
						sb.do_move(n->move);
					}

					if(n->uct_visits > UCTN) {
						TABLE::create_children(&sb,n);
						Node* next = TABLE::UCT_select(n);
						if(next) {
							sb.do_move(next->move);
							n = next;
						}
					}

					l_add(n->workers,1);
				}

				//playout the position
				b.copy(sb);
				cache[threadId] = b.playout(sb);

				//update result
				if(threadIdWarp == 0) {
					l_sub(n->workers,1);

					U32 score = 0;
					for(int i = 0;i < WARP;i++)
						score += cache[threadId + i];
					if(sb.player == 0) 
						score = nLoop * WARP - score;
						
					Node* current = n;
					while(current) {
						l_lock(current->lock);
						current->uct_wins += score;
						current->uct_visits += nLoop * WARP;
						l_unlock(current->lock);
						score = nLoop * WARP - score;
						current = current->parent;
					}
					if(TABLE::root_node->uct_visits >= N)
						finished = true;
				}
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
	cudaThreadSynchronize();
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

static const char *const commands_recognized[] = {
	"d",
	"go",
	"quit",
	"help",
	NULL
};

int main() {
	init_device();

	char str[64];
	BOARD b;
	b.clear();
	
	printf("\nType <help> for a list of commands.\n\n");
	while(true) {
		printf("$: ");
		scanf("%s",&str);
		if(!strcmp(str,"d")) {
			b.print_board();
		} else if(!strcmp(str,"help")) {
			size_t index = 0;
			while (commands_recognized[index]) {
				puts(commands_recognized[index]);
				index++;
			}
		} else if(!strcmp(str,"go")) {
			clock_t start,end;
			start = clock();
			simulate(&b,128 * 2 * 128 * 100);
			end = clock();
			printf("time %d\n",end - start);
		} else if(!strcmp(str,"quit")) {
			break;
		} else {
			MOVE move;
			b.str_mov(move,str);
			b.do_move(move);
		}
	}

	finalize_device();
}

//
// end
//

