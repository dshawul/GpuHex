#include <string>
#include <ctime>
#include <cstdio>
#include "hex.h"

using namespace std;

void print_bitboard(U64 b);

#ifndef GPU
//
// CPU code for monte carlo simulation. It is out of sync with the cuda
// code because it is rarely updated.
//
U32 nLoop;

void BOARD::seed(int sd) {
	randn = sd;
}
U32 BOARD::rand() {
	randn *= 214013;
	randn += 2531011;
	return ((randn >> 16) & 0x7fff);
}
U32 BOARD::playout() {
	U64 wpawns_;
	U64 all_;
	char player_;
	char emptyc_;

	wpawns_ = wpawns;
	all_ = all;
	player_ = player;
	emptyc_ = emptyc;

	U32 wins = 0;
	for(U32 i = 0;i < nLoop;i++) {
		
		wpawns = wpawns_;
		all = all_;
		player = player_;
		emptyc = emptyc_;

		for(;emptyc > 0;emptyc--) {
			U32 rbit = rand() % emptyc;
			U64 mbit = all;
			for(U32 i = 0;i < rbit;i++)
				mbit &= mbit - 1; 
			mbit = mbit & -mbit;

			if(player == 0)
				wpawns ^= mbit;
			all ^= mbit;
			player ^= 1;
		}

		U64 m = (wpawns & UINT64(0x00000000000000ff)),oldm;
		do {
			oldm = m;
			m |=((((m << 8) | (m >> 8)) | 
				 (((m << 9) | (m << 1)) & UINT64(0xfefefefefefefefe)) | 
				 (((m >> 9) | (m >> 1)) & UINT64(0x7f7f7f7f7f7f7f7f))) 
				 & wpawns
				);
			if(m & UINT64(0xff00000000000000)) {
				wins++;
				break;
			}
		} while(m != oldm);
	}
	return wins;
}
void simulate(BOARD* b,U32 N) {
	b->seed(0);
	nLoop = N;
	U32 wins = b->playout();
	printf("%u %u %.6f\n",wins,N,float(wins)/N);
}
void init_device() {
}
void finalize_device() {
}
#endif

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
	const U32 nSimulations = 8 * 14 * 256 * 1600; 

	init_device();

	BOARD b;
	b.init();
    clock_t start,end;
	start = clock();
	simulate(&b,nSimulations);
	end = clock();
	printf("time %d\n",end - start);

	finalize_device();
}


