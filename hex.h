#ifndef __HEX_H
#define __HEX_H

//#define GPU

typedef unsigned __int64 U64;
typedef unsigned int U32;

#define UINT64(x) (x##ui64)

struct BOARD {
	U64 wpawns;
	U64 all;
	U32 randn;
	char player;
	char emptyc;

	void seed(int sd);
	U32 rand();

	U32 playout();
	void make_random_move();
	bool is_white_win();

	void init() {
		wpawns = 0;
		all = UINT64(0xffffffffffffffff);
		emptyc = 64;
		player = 0;
	}
	
};
void simulate(BOARD* b,U32 N);
void init_device();
void finalize_device();

#endif