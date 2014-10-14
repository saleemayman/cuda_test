#ifndef __CLASSES__H_
#define __CLASSES__H_

#include <stdlib.h>
#include <stdio.h>
#include <string>

//template <typename T>
class CMain
{
//private:
public:
	int *a, *b, *c;
	int size_h;

//public:
	// constructor
	CMain(int _size);

	// destructor
	~CMain();

	// initialize an array
	void initialize();
	void addVectors();
	void resultPrint();
	void memAlloc();
	void memRelease();
};

#endif
