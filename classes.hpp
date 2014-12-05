#ifndef __CLASSES__H_
#define __CLASSES__H_

//#include "classes.hpp"
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <vector>

class CMain
{
public:
	long int *a, *b, *c;
	size_t size_h;

	std::vector<void *> arguments;

	// constructor
	CMain(size_t _size)
	{
		size_h = _size;
	}

	// destructor
	~CMain()
	{}

	// initialize an array
	void initialize()
	{
		// intialize array based on type
		for (size_t	i = 0; i < size_h; i++)
		{
			a[i] = i;
			b[i] = (i % 5) + 1;
			c[i] = 0;
		}
	}

	void addVectors()
	{
		for (size_t i = 0; i < size_h; i++)
		{
			c[i] = a[i] + b[i];
		}
	}

	void resultPrint()
	{
		for(size_t i = 0; i < size_h; i++)
		{
			//printf("%i: \t %i + %i = %i \n", i, a[i], b[i], c[i]);
		}
		printf("c[%lu]: %lu\n", (size_h - 1), c[size_h - 1]);
	}

	void memAlloc()
	{
		a = new long int[size_h];
		b = new long int[size_h];
		c = new long int[size_h];
	}

	void memRelease()
	{
		delete[] a;
		delete[] b;
		delete[] c;
	}
};


#endif //__CLASSES__H_