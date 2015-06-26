#include "occa.hpp"

#include <stdio.h>

#include "my_types.h"

occa::device device;
occa::kernel playout;
occa::kernel print_tree;
occa::kernel reset_table;
occa::memory hmem_;
occa::memory tab_;
	
//
// Re-define some structs for the CPU
//

//Tree node
typedef struct tagNode {
	MOVE move;
	U32 uct_wins;
	U32 uct_visits;
	int workers;
	struct tagNode* parent;
	struct tagNode* child;
	struct tagNode* next;
	int lock;
} Node;

//MCTS transposition table
typedef struct tagTABLE {
	int size;
	int tsize;
	BOARD root_board;
	Node* mem_;
	Node* root_node;
	Node* head;
	int lock;
} TABLE;

void allocate_table(int N) {
	tab_ = device.malloc(sizeof(TABLE));
	hmem_ = device.malloc(N*sizeof(Node));
}

void init_device() {
	printf("nBlocks=%d X nThreads=%d\n",nBlocks,nThreads);
	allocate_table(TT_SIZE);
}

//
// Do specify number of playouts on board
//
void simulate(BOARD* b,U32 N) {
	TABLE tab;
	tab.root_board = *b;
    tab_.copyFrom(&tab);
	reset_table(tab_,hmem_,N);
	device.finish();
	playout(tab_,N);
	device.finish();
	print_tree(tab_,1);
	device.finish();
}

//
// Test our implementation
//
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

const char* str_sq(int& sq,const char* is) {
	const char* s = is;
	int f = tolower(*s++) - 'a';
	int r = atoi(s++) - 1;
	sq = SQ(r,f);
	return s;
}

static const char *const commands[] = {
	"d",
	"go",
	"quit",
	"help",
	NULL
};

int main(int argc, char** argv) {
	device.setup("mode = CUDA, platformID = 0, deviceID = 0");
	
	init_device();

	playout = device.buildKernelFromSource("hex.occa","playout");
	print_tree = device.buildKernelFromSource("hex.occa","print_tree");
	reset_table = device.buildKernelFromSource("hex.occa","reset_table");
	
 	playout.setWorkingDims(1, nThreads, nBlocks);
	reset_table.setWorkingDims(1, 1, 1);
    print_tree.setWorkingDims(1, 1, 1);
	
	char str[64];
	BOARD b;
	clear_board(b);

	printf("\nType <help> for a list of commands.\n\n");
	while(true) {
		printf("$: ");
		scanf("%s",&str[0]);
		if(!strcmp(str,"d")) {
			print_bitboard(b.wpawns);
			print_bitboard(b.all);
		} else if(!strcmp(str,"help")) {
			size_t index = 0;
			while (commands[index]) {
				puts(commands[index]);
				index++;
			}
		} else if(!strcmp(str,"go")) {
			clock_t start,end;
			start = clock();
			simulate(&b,128 * 28 * 128 * 8);
			end = clock();
			printf("time %lu\n",end - start);
		} else if(!strcmp(str,"quit")) {
			break;
		} else {
			int move;
			str_sq(move,str);
			do_move(b, (U64(1) << move));
		}
	}

	return 0;
}
