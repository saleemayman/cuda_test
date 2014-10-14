#include "classes.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>
using namespace std;

// constructor
CMain::CMain(int _size)
{
	size_h = _size;
}

// destructor
CMain::~CMain(){}

// initialize an array
void CMain::initialize()
{
	// intialize array based on type
	for (int i = 0; i < size_h; i++)
	{
		a[i] = i;
		b[i] = (i % 5) + 1;
		c[i] = 0;
	}
}

void CMain::addVectors()
{
	for (int i = 0; i < size_h; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void CMain::resultPrint()
{
	for(int i = 0; i < size_h; i++)
	{
		printf("%i: \t %i + %i = %i \n", i, a[i], b[i], c[i]);
	}
}

void CMain::memAlloc()
{
	a = new int[size_h];
	b = new int[size_h];
	c = new int[size_h];
}

void CMain::memRelease()
{
	delete[] a;
	delete[] b;
	delete[] c;
}
