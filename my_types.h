//Integer types
#define U64(x) (x##ull)
#define FMTU64 "0x%016llx"
typedef unsigned long long U64;
typedef unsigned int U32;

//file/rank
#define file(x)       ((x) & 7)
#define rank(x)       ((x) >> 3)
#define SQ(x,y)       (((x) << 3) + (y))

//Define block and warp sizes
#define GPU
#ifdef GPU
#	define nThreads   64
#	define nBlocks   112
#	define WARP       32
#else
#	define nThreads    1
#	define nBlocks     1
#	define WARP        1
#endif

#define nLoop     16
#define nWarps    (nThreads / WARP)

//Transposition table size
#define TT_SIZE   4194304

//Move type
typedef U64 MOVE;

//Game board
typedef struct tagBOARD {
	U64 wpawns;
	U64 all;
	U32 randn;
	U32 player;
	U32 emptyc;
} BOARD;

//Board functions
#define clear_board(b) {				\
	b.wpawns = 0;						\
	b.all = U64(0xffffffffffffffff);	\
	b.emptyc = 64;						\
	b.player = 0;						\
}

#define copy_board(b,c) {	\
	b.wpawns = c.wpawns;	\
	b.all = c.all;			\
	b.player = c.player;	\
	b.emptyc = c.emptyc;	\
}

#define do_move(b,move) {	\
	b.all ^= move;			\
	if(b.player == 0)		\
		b.wpawns ^= move;	\
	b.player ^= 1;			\
	b.emptyc--;				\
}

#define seed(b,sd) 	b.randn = (sd);

#define rand(b,v) {						\
	b.randn *= 214013;					\
	b.randn += 2531011;					\
	v = ((b.randn >> 16) & 0x7fff);		\
}

#define make_random_move(b) {		\
	U32 rnd;						\
	rand(b,rnd);					\
	U32 rbit = rnd % b.emptyc;	\
	U64 mbit = b.all;				\
	for(U32 i = 0;i < rbit;i++)		\
		mbit &= mbit - 1;			\
	mbit = mbit & -mbit;			\
	if(b.player == 0)				\
		b.wpawns ^= mbit;			\
	b.all ^= mbit;					\
	b.player ^= 1;					\
	b.emptyc--;						\
}

#define is_white_win(b,v) {											\
	U64 m = (b.wpawns & U64(0x00000000000000ff)),oldm;				\
	v = false;														\
	do {															\
		oldm = m;													\
		m |=((((m << 8) | (m >> 8)) |								\
			 (((m << 9) | (m << 1)) & U64(0xfefefefefefefefe)) |	\
			 (((m >> 9) | (m >> 1)) & U64(0x7f7f7f7f7f7f7f7f)))		\
			 & b.wpawns												\
			);														\
		if(m & U64(0xff00000000000000)) {							\
			v = true; break;										\
		}															\
	} while(m != oldm);												\
}

#define playoutb(b,c,v) {				\
	U32 wins = 0;						\
	for(U32 i = 0;i < nLoop;i++) {		\
		copy_board(b,c);				\
		while(b.emptyc > 0)				\
			make_random_move(b);		\
		bool is_win;					\
		is_white_win(b,is_win);			\
		if(is_win) wins++;				\
	}									\
	v = wins;							\
}
