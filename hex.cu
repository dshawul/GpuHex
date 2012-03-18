
#define GPU __CUDACC__

#define CHESS

#ifdef GPU
#	if __CUDA_ARCH__ >= 200 
#		define HAS_INTRINSIC
#	endif
#endif

#ifdef GPU
#	include <cuda.h>
#else
#	include <omp.h>
#	include <math.h>
#	include <ctime>
#endif
#include <string>

//
// parameters
//
#ifdef GPU
#	ifdef CHESS
#		define nThreads  128
#		define nBlocks    14
#	else
#		define nThreads  512
#		define nBlocks    14
#	endif
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
#	if __CUDA_ARCH__ < 200 
#		include "cuPrintf.cu"
#		define print(format, ...) cuPrintf(format, __VA_ARGS__)
#	else
#		define print(format, ...) printf(format, __VA_ARGS__)
#	endif
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

#define unitBB(x) ((U64)1 << (x))

//
// Collection of bitboard tricks suitable for GPUs from
// http://chessprogramming.wikispaces.com/Bitboards
// Thanks to Gerd Isinberg,Harald Luben and others.
//
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

#ifdef HAS_INTRINSIC
#	define firstone(bb) __ffsll(bb)
#	define popcnt(bb)   __popcll(bb)
#else
__constant__
int index64[64];

__device__
int firstone(U64 bb) {
	unsigned int folded;
	bb ^= bb - 1;
	folded = (int) bb ^ (bb >> 32);
	return index64[folded * 0x78291ACF >> 26];
}

__device__
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
#endif

__device__
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

__device__
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

__device__
U64 diagonalAttacks( U64 occ, int sq ) { 
   union { U64 b; struct { unsigned int l; unsigned int h; }; } b;
   b.b = occ & diagonal_mask_ex[sq]; 
   b.h = ((b.l + b.h) * 0x02020202) >> 26; 
   b.h = b.l = 0x01010101 * first_rank_attacks[b.h][sq&7]; 
   return b.b & diagonal_mask_ex[sq]; 
} 

__device__
U64 antidiagAttacks( U64 occ, int sq ) { 
   union { U64 b; struct { unsigned int l; unsigned int h; }; } b;
   b.b = occ & antidiag_mask_ex[sq]; 
   b.h = ((b.l + b.h) * 0x02020202) >> 26; 
   b.h = b.l = 0x01010101 * first_rank_attacks[b.h][sq&7]; 
   return b.b & antidiag_mask_ex[sq]; 
} 

#define nfileHBB U64(0xfefefefefefefefe)
#define nfileABB U64(0x7f7f7f7f7f7f7f7f)

#define kingAttacks(sq) king_attacks[sq]
#define knightAttacks(sq) knight_attacks[sq]
#define rookAttacks(occ,sq) (fileAttacks(occ,sq) | rankAttacks(occ,sq))
#define bishopAttacks(occ,sq) (diagonalAttacks(occ,sq) | antidiagAttacks(occ,sq))
#define queenAttacks(occ,sq) (rookAttacks(occ,sq) | bishopAttacks(occ,sq))

//
// A linear congruential(LCG) pseudo random number generator (PRNG).
// Microsoft visual c++ constants are used.
//

#define MY_RAND_MAX  0x7fff

struct PRNG {
	U32 randn;

	__device__ __host__
	void seed(int sd) {
		randn = sd;
	}

	__device__ __host__
	U32 rand() {
		randn *= 214013;
		randn += 2531011;
		return ((randn >> 16) & MY_RAND_MAX);
	}

	__device__ __host__
	U64 rand64() {
		return((U64)rand()) ^ 
			  ((U64)rand() << 15) ^ ((U64)rand() << 30) ^
			  ((U64)rand() << 45) ^ ((U64)rand() << 60);
	}
};

//
//sq_to_string and vice versa
//

#define file(x)          ((x) & 7)
#define rank(x)          ((x) >> 3)
#define SQ(x,y)          (((x) << 3) + (y))

__device__ __host__
char* sq_str(const int& sq,char* s) {
	*s++ = 'a' + file(sq);
	*s++ = '1' + rank(sq);
	*s = 0;
	return s;
}

__host__
void str_sq(const char* s,int& sq) {
	int f = tolower(*s++) - 'a';
	int r = atoi(s++) - 1;
	sq = SQ(r,f);
}

__host__
void print_bitboard(U64 b) {
	std::string s = "";
	for(int i = 7;i >= 0;i--) {
		for(int z = 0; z < 7 - i; z++)
			s += " ";
		for(int j = 0;j < 8;j++) {
			U64 m = unitBB(i * 8 + j);
			if(b & m) s += "1 ";
			else s += "0 ";
		}
		s += "\n";
	}
	printf("%s",s.c_str());
	printf("\n"FMTU64"\n\n",b);
}

//
//HEX
//

#ifndef CHESS

typedef U64 MOVE;

#define is_same(x,y) ((x) == (y))

struct BOARD : public PRNG {
	char player;
	char emptyc;
	U64 wpawns;
	U64 all;

	__device__ __host__ 
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
	bool is_legal(const MOVE& move) {
		return true;
	}

	__host__
	void init() {
		wpawns = 0;
		all = U64(0xffffffffffffffff);
		emptyc = 64;
		player = 0;
	}

	__host__
	void print_board() const {
		print_bitboard(wpawns);
		print_bitboard(all);
	}

	U32 playout(const BOARD&);
	bool make_random_move();
	bool is_white_win();

	int count_moves() const;
	MOVE gen_move(const int&) const;
	
	void str_mov(const char*,MOVE& move);
};

__device__
bool BOARD::make_random_move() {
	if(emptyc <= 0) 
		return false;

	U32 rbit = rand() % emptyc;
	U64 move = all;
	for(U32 i = 0;i < rbit;i++)
		move &= move - 1;
	move = move & -move;
	do_move(move);

	return true;
}

__device__
bool BOARD::is_white_win(){
	U64 m = (wpawns & U64(0x00000000000000ff)),oldm;
	do {
		oldm = m;
		m |=((((m << 8) | (m >> 8)) | 
			 (((m << 9) | (m << 1)) & nfileHBB) | 
			 (((m >> 9) | (m >> 1)) & nfileABB)) 
			 & wpawns
			);
		if(m & U64(0xff00000000000000))
			return true;
	} while(m != oldm);
	return false;
}

__device__
U32 BOARD::playout(const BOARD& b) {
	U32 wins = 0;
	for(U32 i = 0;i < nLoop;i++) {
		copy(b);

		while(make_random_move())
			;
			
		if(is_white_win())
			wins++;
	}
	return wins;
}

__device__
int BOARD::count_moves() const {
	return emptyc;
}

__device__
MOVE BOARD::gen_move(const int& index) const {
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

__device__
void mov_str(const MOVE& move,char* s) {
	int sq = firstone(move);
	sq_str(sq,s);
}
__host__
void BOARD::str_mov(const char* s,MOVE& move) {
	int sq;
	str_sq(s,sq);
	move = unitBB(sq);
}

//
// CHESS
//

#else

__constant__ char d_piece_name[] = "__KQRBNP";
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
enum SQUARES {
	A1 = 0,B1,C1,D1,E1,F1,G1,H1,
	A8 =56,B8,C8,D8,E8,F8,G8,H8
};


//Move is 32 bit
typedef U32 MOVE;

#define FROM_FLAG        0x000000ff
#define TO_FLAG          0x0000ff00
#define PIECE_FLAG       0x000f0000
#define PROMOTION_FLAG   0x00f00000
#define CAPTURE_FLAG     0x0f000000
#define EP_FLAG          0x10000000
#define CASTLE_FLAG      0x20000000
#define FROM_TO          0x0000ffff
#define is_ep(x)         ((x) & EP_FLAG)
#define is_castle(x)     ((x) & CASTLE_FLAG)
#define is_same(x,y)     (((x) & FROM_TO) == ((y) & FROM_TO))
#define m_from(x)        (int)((x) & FROM_FLAG)
#define m_to(x)          (int)(((x) & TO_FLAG) >> 8)
#define m_piece(x)       (int)(((x) & PIECE_FLAG) >> 16)
#define m_promote(x)     (int)(((x) & PROMOTION_FLAG) >> 20)
#define m_capture(x)     (int)(((x) & CAPTURE_FLAG) >> 24)
#define m_make(from,to,pic,prom,flag) \
	((from) | ((to) << 8) | ((pic) << 16) | ((prom) << 20) | (flag))

//BOARD
struct BOARD : public PRNG {
	union {
		struct {
			unsigned char player;
			unsigned char epsquare;
			unsigned char fifty;
			unsigned char castle;
		};
		U32 flags;
	};
	U64 wpieces;
	U64 bpieces;
	U64 kings;
	U64 queens;
	U64 rooks;
	U64 bishops;
	U64 knights;
	U64 pawns;

	__device__ __host__ 
	void copy(const BOARD& b) {
		wpieces = b.wpieces;
		bpieces = b.bpieces;
		kings = b.kings;
		queens = b.queens;
		rooks = b.rooks;
		bishops = b.bishops;
		knights = b.knights;
		pawns = b.pawns;
		flags = b.flags;
	}

	__host__
	void init() {
		set_fen(start_fen);
	}

	__host__
	void print_board() const{
		char fen[256];
		get_fen(fen);
		printf("%s\n",fen);
	}

	bool is_legal(MOVE&);
	U32  playout(const BOARD&);
	bool make_random_move();
	bool is_white_win();

	void do_move(MOVE& move);
	void undo_move(const MOVE& move);
	int count_moves() const;
	MOVE gen_move(const int&) const;
	bool attacks(const U64&,int,int);

	void set_fen(const char* fen);
	void get_fen(char* fen) const;
	void str_mov(const char*,MOVE&);
};

__device__ __host__
void BOARD::do_move(MOVE& move) {
	const int from = m_from(move);
	const int to = m_to(move);
	const U64 mfromBB = unitBB(from);
	const U64 mtoBB = unitBB(to);

	//capture
	bool is_capture = false;
	if(is_ep(move)) {
		if(player == white) {
			bpieces ^= unitBB(to - 8);
			pawns ^= unitBB(to - 8);
		} else {
			wpieces ^= unitBB(to + 8);
			pawns ^= unitBB(to + 8);
		}
		is_capture = true;
	} else {
		U64 toBB;
		if(player == white) {
			toBB = mtoBB & bpieces;
			if(toBB) bpieces ^= mtoBB;
		} else {
			toBB = mtoBB & wpieces;
			if(toBB) wpieces ^= mtoBB;
		}
		if(toBB) {
			if(toBB & pawns) { 
				pawns ^= mtoBB; 
				move |= (pawn << 24); 
			} else if(toBB & knights) { 
				knights ^= mtoBB; 
				move |= (knight << 24); 
			} else if(toBB & bishops) { 
				bishops ^= mtoBB; 
				move |= (bishop << 24); 
			} else if(toBB & rooks) { 
				rooks ^= mtoBB; 
				move |= (rook << 24); 
			} else if(toBB & queens) { 
				queens ^= mtoBB; 
				move |= (queen << 24); 
			}
			is_capture = true;
		}
	}

	//erase from and set to
	{
		if(player == white) {
			wpieces ^= mfromBB;
			wpieces ^= mtoBB;
		} else {
			bpieces ^= mfromBB;
			bpieces ^= mtoBB;
		}

		int pic = m_piece(move);
		switch(pic) {
			case king: kings ^= mfromBB; break;
			case queen: queens ^= mfromBB; break;
			case rook: rooks ^= mfromBB; break;
			case bishop: bishops ^= mfromBB; break;
			case knight: knights ^= mfromBB; break;
			case pawn: pawns ^= mfromBB; break;
		}
		if(m_promote(move)) 
			pic = m_promote(move);
		switch(pic) {
			case king: kings ^= mtoBB; break;
			case queen: queens ^= mtoBB; break;
			case rook: rooks ^= mtoBB; break;
			case bishop: bishops ^= mtoBB; break;
			case knight: knights ^= mtoBB; break;
			case pawn: pawns ^= mtoBB; break;
		}
	}

	//castle
	if(is_castle(move)) {
        int fromc,toc;
		if(to > from) {
           fromc = to + 1;
		   toc = to - 1;
		} else {
           fromc = to - 2;
		   toc = to + 1;
		}
		
		if(player == white) {
			wpieces ^= unitBB(fromc);
			wpieces ^= unitBB(toc);
		} else {
			bpieces ^= unitBB(fromc);
			bpieces ^= unitBB(toc);
		}
        rooks ^= unitBB(fromc);
		rooks ^= unitBB(toc);
	} 

	//flags
	{
		//fifty
		fifty++;
		if(is_capture)
			fifty = 0;

		//enpassant
		epsquare = 0;
		if(m_piece(move) == pawn) {
			fifty = 0;
			if(player == white) {
				if((to == from + 16) && (rank(from) == RANK2))
					epsquare = (from + to) >> 1;
			} else {
				if((to == from - 16) && (rank(from) == RANK7))
					epsquare = (from + to) >> 1;
			}
		}

		//castle
		if(castle) {
			if(from == E1 || to == A1 || from == A1) castle &= ~WLC_FLAG;
			if(from == E1 || to == H1 || from == H1) castle &= ~WSC_FLAG;
			if(from == E8 || to == A8 || from == A8) castle &= ~BLC_FLAG;
			if(from == E8 || to == H8 || from == H8) castle &= ~BSC_FLAG;
		}

		//player
		player ^= 1;
	}
}
__device__ __host__
void BOARD::undo_move(const MOVE& move) {
	const int from = m_from(move);
	const int to = m_to(move);
	const U64 mfromBB = unitBB(from);
	const U64 mtoBB = unitBB(to);
	int pic;

	//erase from and set to
	{
		if(player == white) {
			wpieces ^= mfromBB;
			wpieces ^= mtoBB;
		} else {
			bpieces ^= mfromBB;
			bpieces ^= mtoBB;
		}

		pic = m_piece(move);
		switch(pic) {
			case king: kings ^= mfromBB; break;
			case queen: queens ^= mfromBB; break;
			case rook: rooks ^= mfromBB; break;
			case bishop: bishops ^= mfromBB; break;
			case knight: knights ^= mfromBB; break;
			case pawn: pawns ^= mfromBB; break;
		}
		if(m_promote(move)) 
			pic = m_promote(move);
		switch(pic) {
			case king: kings ^= mtoBB; break;
			case queen: queens ^= mtoBB; break;
			case rook: rooks ^= mtoBB; break;
			case bishop: bishops ^= mtoBB; break;
			case knight: knights ^= mtoBB; break;
			case pawn: pawns ^= mtoBB; break;
		}
	}

	//castle
	if(is_castle(move)) {
        int fromc,toc;
		if(to > from) {
           fromc = to + 1;
		   toc = to - 1;
		} else {
           fromc = to - 2;
		   toc = to + 1;
		}
		
		if(player == white) {
			wpieces ^= unitBB(fromc);
			wpieces ^= unitBB(toc);
		} else {
			bpieces ^= unitBB(fromc);
			bpieces ^= unitBB(toc);
		}
        rooks ^= unitBB(fromc);
		rooks ^= unitBB(toc);
	} 

	//capture
	if(is_ep(move)) {
		if(player == white) {
			bpieces ^= unitBB(to - 8);
			pawns ^= unitBB(to - 8);
		} else {
			wpieces ^= unitBB(to + 8);
			pawns ^= unitBB(to + 8);
		}
	} else if(pic = m_capture(move)) {
		if(player == white)
			bpieces ^= mtoBB;
		else
			wpieces ^= mtoBB;
		switch(pic) {
			case king: kings ^= mtoBB; break;
			case queen: queens ^= mtoBB; break;
			case rook: rooks ^= mtoBB; break;
			case bishop: bishops ^= mtoBB; break;
			case knight: knights ^= mtoBB; break;
			case pawn: pawns ^= mtoBB; break;
		}
	}
}
__device__
int BOARD::count_moves() const {
	const U64 occ = wpieces | bpieces;
	U64 pieceb,movesb;
	int from,count = 0;

	if(player == white) {

		//non-promotions
		pieceb = wpieces & pawns;
		pieceb &= U64(0x0000ffffffffff00);

		movesb = (pieceb << 8) & ~occ;
		movesb |= ((movesb << 8) & ~occ) & U64(0x00000000ff000000);
		count += popcnt(movesb);

		movesb = ((pieceb & nfileHBB) << 7) & bpieces;
		count += popcnt(movesb);

		movesb = ((pieceb & nfileABB) << 9) & bpieces;
		count += popcnt(movesb);

		if(epsquare) {
			from = epsquare - 9;
			if((file(from) < FILEH) && (unitBB(from) & pieceb))
				count++;
			from = epsquare - 7;
			if((file(from) > FILEA) && (unitBB(from) & pieceb))
				count++;
		}

		//promotions
		pieceb = wpieces & pawns;
		pieceb &= U64(0x00ff000000000000);

		movesb = (pieceb << 8) & ~occ;
		count += 4 * popcnt(movesb);

		movesb = ((pieceb & nfileHBB) << 7) & bpieces;
		count += 4 * popcnt(movesb);

		movesb = ((pieceb & nfileABB) << 9) & bpieces;
		count += 4 * popcnt(movesb);

		//legal board
		const U64 legalB = (~occ | bpieces);

		//knight
		pieceb = wpieces & knights;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = knightAttacks(from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//bishop
		pieceb = wpieces & bishops;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = bishopAttacks(occ,from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//rook
		pieceb = wpieces & rooks;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = rookAttacks(occ,from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//queen
		pieceb = wpieces & queens;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = queenAttacks(occ,from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//king
		pieceb = wpieces & kings;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = kingAttacks(from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//castle
		if(castle & WSLC_FLAG) {
			if((castle & WSC_FLAG) &&
				!(occ & U64(0x0000000000000060))
				) {
					count++;
			}
			if((castle & WLC_FLAG) &&
				!(occ & U64(0x000000000000000e))
				){
					count++;
			}
		}
	} else {

		//non-promotions
		pieceb = bpieces & pawns;
		pieceb &= U64(0x00ffffffffff0000);
		movesb = (pieceb >> 8) & ~occ;
		movesb |= ((movesb >> 8) & ~occ) & U64(0x000000ff00000000);
		count += popcnt(movesb);

		movesb = ((pieceb & nfileABB) >> 7) & wpieces;
		count += popcnt(movesb);

		movesb = ((pieceb & nfileHBB) >> 9) & wpieces;
		count += popcnt(movesb);

		if(epsquare) {
			from = epsquare + 9;
			if((file(from) > FILEA) && (unitBB(from) & pieceb))
				count++;
			from = epsquare + 7;
			if((file(from) < FILEH) && (unitBB(from) & pieceb))
				count++;
		}

		//promotions
		pieceb = bpieces & pawns;
		pieceb &= U64(0x000000000000ff00);
		movesb = (pieceb >> 8) & ~occ;
		count += 4 * popcnt(movesb);

		movesb = ((pieceb & nfileABB) >> 7) & wpieces;
		count += 4 * popcnt(movesb);

		movesb = ((pieceb & nfileHBB) >> 9) & wpieces;
		count += 4 * popcnt(movesb);

		//legal board
		const U64 legalB = (~occ | wpieces);

		//knight
		pieceb = bpieces & knights;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = knightAttacks(from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//bishop
		pieceb = bpieces & bishops;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = bishopAttacks(occ,from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//rook
		pieceb = bpieces & rooks;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = rookAttacks(occ,from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//queen
		pieceb = bpieces & queens;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = queenAttacks(occ,from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//king
		pieceb = bpieces & kings;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = kingAttacks(from) & legalB;
			count += popcnt(movesb);
			pieceb &= pieceb - 1;
		}

		//castle
		if(castle & BSLC_FLAG) {
			if((castle & BSC_FLAG) &&
				!(occ & U64(0x6000000000000000))
				) {
					count++;
			}
			if((castle & BLC_FLAG) &&
				!(occ & U64(0x0e00000000000000))
				){
					count++;
			}
		}
	}
	return count;
} 
__device__
MOVE BOARD::gen_move(const int& index) const {
	const U64 occ = wpieces | bpieces;
	U64 pieceb,movesb;
	int from,to,count = 0;

	//
	//MOVE
	//
#define MOVEPC(pic,prom,flag) {							\
	count++;											\
	if(count == index + 1) {							\
		MOVE move = m_make(from,to,pic,prom,flag);		\
		return move;									\
	}													\
};

#define MOVEP(pic)				MOVEPC(pic,0,0)
#define MOVEP_EP(pic)			MOVEPC(pic,0,EP_FLAG)
#define MOVEP_CASTLE(pic)		MOVEPC(pic,0,CASTLE_FLAG)
#define MOVEP_PROMOTE(pic)  {	\
	MOVEPC(pic,queen,0);		\
	MOVEPC(pic,knight,0);		\
	MOVEPC(pic,rook,0);			\
	MOVEPC(pic,bishop,0);		\
};

	if(player == white) {

		//non-promotions
		pieceb = wpieces & pawns;
		pieceb &= U64(0x0000ffffffffff00);
		movesb = (pieceb << 8) & ~occ;
		U64 tempb = movesb;
		while(movesb) {
			to = firstone(movesb);
			from = to - 8;	
			MOVEP(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((tempb << 8) & ~occ) & U64(0x00000000ff000000);
		while(movesb) {
			to = firstone(movesb);
			from = to - 16;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileHBB) << 7) & bpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to - 7;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileABB) << 9) & bpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to - 9;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}
		//enpassant
		if(epsquare) {
			to = epsquare;
			from = epsquare - 9;
			if((file(from) < FILEH) && (unitBB(from) & pieceb))
				MOVEP_EP(pawn);
			from = epsquare - 7;
			if((file(from) > FILEA) && (unitBB(from) & pieceb))
				MOVEP_EP(pawn);
		}
		//promote
		pieceb = wpieces & pawns;
		pieceb &= U64(0x00ff000000000000);
		movesb = (pieceb << 8) & ~occ;
		while(movesb) {
			to = firstone(movesb);
			from = to - 8;
			MOVEP_PROMOTE(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileHBB) << 7) & bpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to - 7;
			MOVEP_PROMOTE(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileABB) << 9) & bpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to - 9;
			MOVEP_PROMOTE(pawn);
			movesb &= movesb - 1;
		}

		//legal board
		const U64 legalB = (~occ | bpieces);

		//knight
		pieceb = wpieces & knights;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = knightAttacks(from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(knight);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//bishop
		pieceb = wpieces & bishops;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = bishopAttacks(occ,from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(bishop);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//rook
		pieceb = wpieces & rooks;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = rookAttacks(occ,from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(rook);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//queen
		pieceb = wpieces & queens;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = queenAttacks(occ,from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(queen);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//king
		pieceb = wpieces & kings;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = kingAttacks(from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(king);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//castle
		if(castle & WSLC_FLAG) {
			if((castle & WSC_FLAG) &&
				!(occ & U64(0x0000000000000060))
				) {
					from = E1;
					to = G1;
					MOVEP_CASTLE(king);
			}
			if((castle & WLC_FLAG) &&
				!(occ & U64(0x000000000000000e))
				){
					from = E1;
					to = C1;
					MOVEP_CASTLE(king);
			}
		}
		
	} else {

		//non-promotions
		pieceb = bpieces & pawns;
		pieceb &= U64(0x00ffffffffff0000);
		movesb = (pieceb >> 8) & ~occ;
		U64 tempb = movesb;
		while(movesb) {
			to= firstone(movesb);
			from = to + 8;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((tempb >> 8) & ~occ) & U64(0x000000ff00000000);
		while(movesb) {
			to = firstone(movesb);
			from = to + 16;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileABB) >> 7) & wpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to + 7;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileHBB) >> 9) & wpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to + 9;
			MOVEP(pawn);
			movesb &= movesb - 1;
		}
		//enpassant
		if(epsquare) {
			to = epsquare;
			from = epsquare + 9;
			if((file(from) > FILEA) && (unitBB(from) & pieceb))
				MOVEP_EP(pawn);
			from = epsquare + 7;
			if((file(from) < FILEH) && (unitBB(from) & pieceb))
				MOVEP_EP(pawn);
		}
		//promote
		pieceb = bpieces & pawns;
		pieceb &= U64(0x000000000000ff00);
		movesb = (pieceb >> 8) & ~occ;
		while(movesb) {
			to= firstone(movesb);
			from = to + 8;
			MOVEP_PROMOTE(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileABB) >> 7) & wpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to + 7;
			MOVEP_PROMOTE(pawn);
			movesb &= movesb - 1;
		}

		movesb = ((pieceb & nfileHBB) >> 9) & wpieces;
		while(movesb) {
			to = firstone(movesb);
			from = to + 9;
			MOVEP_PROMOTE(pawn);
			movesb &= movesb - 1;
		}

		//legal board
		const U64 legalB = (~occ | wpieces);

		//knight
		pieceb = bpieces & knights;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = knightAttacks(from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(knight);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//bishop
		pieceb = bpieces & bishops;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = bishopAttacks(occ,from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(bishop);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//rook
		pieceb = bpieces & rooks;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = rookAttacks(occ,from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(rook);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//queen
		pieceb = bpieces & queens;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = queenAttacks(occ,from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(queen);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//king
		pieceb = bpieces & kings;
		while(pieceb) {
			from = firstone(pieceb);
			movesb = kingAttacks(from) & legalB;

			while(movesb) {
				to = firstone(movesb);
				MOVEP(king);
				movesb &= movesb - 1;
			}

			pieceb &= pieceb - 1;
		}

		//castle
		if(castle & BSLC_FLAG) {
			if((castle & BSC_FLAG) &&
				!(occ & U64(0x6000000000000000))
				) {
					from = E8;
					to = G8;
					MOVEP_CASTLE(king);
			}
			if((castle & BLC_FLAG) &&
				!(occ & U64(0x0e00000000000000))
				){
					from = E8;
					to = C8;
					MOVEP_CASTLE(king);
			}
		}
	}

	return MOVE();
}

__device__
bool BOARD::attacks(const U64& occ,int sq,int col) {
	if(col == white) {
		U64 wpawns = wpieces & pawns;
		if(file(sq) > FILEA && (unitBB(sq - 9) & wpawns)) return true;
		if(file(sq) < FILEH && (unitBB(sq - 7) & wpawns)) return true;
		if(knightAttacks(sq) & wpieces & knights) return true;
		if(kingAttacks(sq) & wpieces & kings) return true;
		U64 bishopsQueens = wpieces & (bishops | queens );
		if(bishopAttacks(occ, sq) & bishopsQueens) return true;
		U64 rooksQueens = wpieces & (rooks | queens);
		if(rookAttacks (occ, sq) & rooksQueens) return true;
	} else {
		U64 bpawns = bpieces & pawns;
		if(file(sq) > FILEA && (unitBB(sq + 7) & bpawns)) return true;
		if(file(sq) < FILEH && (unitBB(sq + 9) & bpawns)) return true;
		if(knightAttacks(sq) & bpieces & knights) return true;
		if(kingAttacks(sq) & bpieces & kings) return true;
		U64 bishopsQueens = bpieces & (bishops | queens );
		if(bishopAttacks(occ, sq) & bishopsQueens) return true;
		U64 rooksQueens = bpieces & (rooks | queens);
		if(rookAttacks (occ, sq) & rooksQueens) return true;
	}
	return false;
}

__device__
bool BOARD::is_legal(MOVE& move) {
	U32 sflags;

	sflags = flags;
	do_move(move);

	int ksq;
	if(player == white)
		ksq = firstone(bpieces & kings);
	else
		ksq = firstone(wpieces & kings);
	bool is_attacked = attacks(wpieces | bpieces,ksq,player);

	flags = sflags;
	undo_move(move);

	return !is_attacked;
}

struct BITSET {
	U32 b0;
	U32 b1;
	U32 b2;
	U32 b3;
	U32 b4;
	U32 b5;
	U32 b6;
	U32 b7;
	U32 counter;
	
	__device__ __host__
	void clear() {
		b0 = 0;
		b1 = 0;
		b2 = 0;
		b3 = 0;
		b4 = 0;
		b5 = 0;
		b6 = 0;
		b7 = 0;
		counter = 0;
	}

	__device__ __host__
	int setbit(int index) {
		U32 r = (1 << (index & 31));

#define CASE(x) {						\
	case x:								\
		if(b ## x & r) return 0;		\
		else b ## x |= r;				\
		break;							\
};
		switch(index >> 5) {
			CASE(0);
			CASE(1);
			CASE(2);
			CASE(3);
			CASE(4);
			CASE(5);
			CASE(6);
			CASE(7);
		}

		counter++;
		return 1;
	}
};

__device__
bool BOARD::make_random_move() {
	MOVE move;
	U32 N,index,sflags;
	int ksq;

	N = count_moves();
	if(N <= 0) 
		return false;

	BITSET moves;
	moves.clear();

	while(true) {
		index = U32(N * (rand() / float(MY_RAND_MAX)));
		if(!moves.setbit(index)) {
			if(moves.counter == N)
				return false;
			else
				continue;
		}
		move = gen_move(index);

		sflags = flags;
		do_move(move);

		if(player == white)
			ksq = firstone(bpieces & kings);
		else
			ksq = firstone(wpieces & kings);

		if(attacks(wpieces | bpieces,ksq,player)) {
			flags = sflags;
			undo_move(move);
		} else break;
	};

	return true;
}

__device__
bool BOARD::is_white_win(){
	return (rand() & 1);
}

__device__
U32 BOARD::playout(const BOARD& b) {
	U32 wins = 0;
	for(U32 i = 0;i < nLoop;i++) {
		copy(b);

		for(U32 j = 0;j < 12;j++) {
			if(!make_random_move())
				break;
		}

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
	int i,r,f,sq,pic,col;
	const char* p = fen_str,*pfen;
	for(r = RANK8;r >= RANK1; r--) {
		for(f = FILEA;f <= FILEH;f++) {
			sq = SQ(r,f);
			if((pfen = strchr(piece_name,*p)) != 0) {
				pic = int(strchr(piece_name,*pfen) - piece_name);
				col = (pic >= 6);
				pic = (pic % 6) + 2;

				U64 maskBB = unitBB(sq);
				if(col == white) 
					wpieces ^= maskBB;
				else 
					bpieces ^= maskBB;
				switch(pic) {
					case king: kings ^= maskBB; break;
					case queen: queens ^= maskBB; break;
					case rook: rooks ^= maskBB; break;
					case bishop: bishops ^= maskBB; break;
					case knight: knights ^= maskBB; break;
					case pawn: pawns ^= maskBB; break;
				}

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
		epsquare = char(strchr(file_name,*p) - file_name);
		p++;
		epsquare = char(epsquare + 8 * (strchr(rank_name,*p) - rank_name));
		p++;
	}
	
	//fifty & hply
	p++;
	if(*p) {
		int f;
		sscanf(p,"%d",&f);
		fifty = f;
	} else {
		fifty = 0;
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

			if(kings & unitBB(sq)) pic = king;
			else if(queens & unitBB(sq)) pic = queen;
			else if(rooks & unitBB(sq)) pic = rook;
			else if(bishops & unitBB(sq)) pic = bishop;
			else if(knights & unitBB(sq)) pic = knight;
			else if(pawns & unitBB(sq)) pic = pawn;

			if(pic) {
				pic -= 2;
				if(bpieces & unitBB(sq))
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
	sprintf(str," %d 1",fifty);
	strcat(fen,str);
}

__device__
void mov_str(const MOVE& move,char* s) {
	s = sq_str(m_from(move),s);
	s = sq_str(m_to(move),s);
	if(m_promote(move))
		*s++ = d_piece_name[m_promote(move)];
	*s = 0;
}
__host__
void BOARD::str_mov(const char* s,MOVE& move) {
	int from,to;
	str_sq(s,from);
	str_sq(s + 2,to);
	move = m_make(from,to,0,0,0);
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
	__device__ MOVE root_moves[256];
	__device__ MOVE root_move;
	__device__ int  n_root_moves;
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
	
	__device__
	void generate_moves() {
		n_root_moves = root_board.count_moves();
		for(int i = 0;i < n_root_moves;i++)
			root_moves[i] = root_board.gen_move(i);
	}

	__global__
	void check_legal(MOVE move) {
		root_move = MOVE();
		generate_moves();
		for(int i = 0;i < n_root_moves;i++) {
			if(is_same(move,root_moves[i])) {
				root_move = root_moves[i];
				return;
			}
		}
	}

	__global__
	void print_moves() {
		char str[8];
		generate_moves();
		for(int i = 0;i < n_root_moves;i++) {
			mov_str(root_moves[i],str);
			print("%d.%5s\n",i + 1,(const char*)str);
		}
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
						print("%d.%5s %12d %12d %12.6f\n",
							depth,(const char*)str,
							current->uct_wins,
							current->uct_visits,
							current->uct_wins / float(current->uct_visits)
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
			if(!b->is_legal(move))
				continue;
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

		cudaMemcpyToSymbol(knight_attacks,mknight_attacks,sizeof(mknight_attacks));
		cudaMemcpyToSymbol(king_attacks,mking_attacks,sizeof(mking_attacks));
		cudaMemcpyToSymbol(diagonal_mask_ex,mdiagonal_mask_ex,sizeof(mdiagonal_mask_ex));
		cudaMemcpyToSymbol(antidiag_mask_ex,mantidiag_mask_ex,sizeof(mantidiag_mask_ex));
		cudaMemcpyToSymbol(first_rank_attacks,mfirst_rank_attacks,sizeof(mfirst_rank_attacks));
#else
		hmem_ = (Node*) malloc(N * sizeof(Node));
		tsize = N;
		mem_ = hmem_;
		l_create(lock);

		memcpy(knight_attacks,mknight_attacks,sizeof(mknight_attacks));
		memcpy(king_attacks,mking_attacks,sizeof(mking_attacks));
		memcpy(diagonal_mask_ex,mdiagonal_mask_ex,sizeof(mdiagonal_mask_ex));
		memcpy(antidiag_mask_ex,mantidiag_mask_ex,sizeof(mantidiag_mask_ex));
		memcpy(first_rank_attacks,mfirst_rank_attacks,sizeof(mfirst_rank_attacks));
#endif

#ifndef HAS_INTRINSIC
		static const int mindex64[64] = {
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
		cudaMemcpyToSymbol(index64,mindex64,sizeof(mindex64));
#else
		memcpy(index64,mindex64,sizeof(mindex64));
#endif
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
					//swap score
					{
						if(sb.player == 0) 
							score = nLoop * WARP - score;		
					}
					Node* current = n;
					while(current) {
						l_lock(current->lock);
						current->uct_wins += score;
						current->uct_visits += nLoop * WARP;
						l_unlock(current->lock);
						//swap score
						{
							score = nLoop * WARP - score;
						}
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
	cudaMemcpyToSymbol(TABLE::root_board,b,sizeof(BOARD));

	TABLE::reset <<<1,1>>> ();
	playout <<<nBlocks,nThreads>>> (N); 
	cudaThreadSynchronize();
	TABLE::print_tree <<<1,1>>> (1);

	cudaPrintfDisplay();
	printf("Errors: %s\n", 
		cudaGetErrorString(cudaPeekAtLastError()));
}

__host__ 
MOVE get_move(BOARD* b,const char* str) {
	cudaMemcpyToSymbol(TABLE::root_board,b,sizeof(BOARD));

	MOVE move;
	b->str_mov(str,move);
	TABLE::check_legal<<<1,1>>>(move);

	cudaMemcpyFromSymbol(&move,TABLE::root_move,sizeof(MOVE));
	
	return move;
}

__host__ 
void print_moves(BOARD* b) {
	cudaMemcpyToSymbol(TABLE::root_board,b,sizeof(BOARD));

	TABLE::print_moves<<<1,1>>>();

	cudaPrintfDisplay();
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
void simulate(BOARD* b,U32 N) {
	TABLE::root_board = *b;
	TABLE::reset();
	playout(N);
	TABLE::print_tree(1);
}

__host__ 
MOVE get_move(BOARD* b,const char* str) {
	TABLE::root_board = *b;
	MOVE move;
	b->str_mov(str,move);
	TABLE::check_legal(move);
	move = TABLE::root_move;
	return move;
}

__host__ 
void print_moves(BOARD* b) {
	TABLE::root_board = *b;
	TABLE::print_moves();
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

	char str[256];
	BOARD b;
	b.init();

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
			simulate(&b,128 * 1 * 128 * 100);
			end = clock();
			printf("time %d\n",end - start);
		} else if(!strcmp(str,"quit")) {
			printf("Bye bye\n");
			break;
		} else if(!strcmp(str,"moves")) {
			print_moves(&b);
		} else {
			MOVE move = get_move(&b,str);
			if(move != 0)
				b.do_move(move);
			else
				printf("Illegal move\n");
		}
	}

	finalize_device();
}

//
// end
//

