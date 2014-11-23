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
	int *a, *b, *c;
	int size_h;

	std::vector<void *> arguments;

	// constructor
	CMain(int _size)
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
		for (int i = 0; i < size_h; i++)
		{
			a[i] = i;
			b[i] = (i % 5) + 1;
			c[i] = 0;
		}
	}

	void addVectors()
	{
		for (int i = 0; i < size_h; i++)
		{
			c[i] = a[i] + b[i];
		}
	}

	void resultPrint()
	{
		for(int i = 0; i < size_h; i++)
		{
			printf("%i: \t %i + %i = %i \n", i, a[i], b[i], c[i]);
		}
	}

	void memAlloc()
	{
		a = new int[size_h];
		b = new int[size_h];
		c = new int[size_h];
	}

	void memRelease()
	{
		delete[] a;
		delete[] b;
		delete[] c;
	}

// ************** T E S T
	inline void setKernelArg(int index, char *arg)
	{
//		arguments.push_back(arg);
		arguments[index] = arg;
	}

	inline void setKernelArg(int index, int *arg)
	{
//		arguments.push_back(arg);
		arguments[index] = arg;
	}

	inline void setKernelArg(int index, float *arg)
	{
//		arguments.push_back(arg);
		arguments[index] = arg;
	}

	inline void argSet()
	{
		arguments.reserve(5);

		setKernelArg(0, a);
		setKernelArg(1, b);
		setKernelArg(2, c);
		setKernelArg(3, &size_h);

		//printf("--args[0..3]  %i  %i  %i  %i \n", arguments[0], arguments[1], arguments[2], arguments[3] );
	}

	inline void argPrint(const std::vector<void *>& args)
	{
		//printf("argsPrint[0..3]  %i  %i  %i  %i \n", args[0], args[1], args[2], args[3] );
	}

// ************* E N D 	T E S T


};


#endif //__CLASSES__H_