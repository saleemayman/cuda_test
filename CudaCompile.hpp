#ifndef __CUDACOMPILE__H_
#define __CUDACOMPILE__H_

#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string>
#include <sstream>

#include "common.h"

class CProgram
{
private:
	std::string command;	///< command string to execute
	std::stringstream command_string_stream;	///< string to append
	const char *file_name;	///< cuda file to compile

public:
	int error;			///< CUDA error handle
	size_t kernelMaxRegisters;

	inline CProgram(const char *file)
	{
		file_name = file;
	}

	/**
	 * compile program from existing command
	 */
	inline CProgram(	const std::string &command,	// context
						const char *file_name	// source file
	)
	{
		executeCommand(command.c_str(), file_name);
	}

	/**
	 * create compile command and compile
	 */
	inline void createCompileCommand(int compute_capability)
	{
		command_string_stream << "nvcc -x cu -keep ";
		if (kernelMaxRegisters != 0)
		{
			command_string_stream << "-maxrregcount ";
			kernelMaxRegisters = 16;
			command_string_stream << kernelMaxRegisters;
		}

		command_string_stream << " ";
		command_string_stream << "-D DEFAULTS=0 ";
		command_string_stream << "-D DOMAIN_CELLS_X=5 ";
		command_string_stream << "-D DOMAIN_CELLS_Y=1 ";
		command_string_stream << "-D LOCAL_WORK_GROUP_SIZE=";
		command_string_stream << LOCAL_WORK_GROUP_SIZE;
		//command_string_stream << " -D DATA_TYPE=1";
		command_string_stream << " -D TYPE_FLOAT=1 ";
		command_string_stream << " -arch=sm_";
		command_string_stream << compute_capability;
		command_string_stream << "0 -m64 -I. -dc ";
		command_string_stream << file_name;
		command_string_stream << ".cu -o ";
		command_string_stream << file_name;
		command_string_stream << ".o ";

		command = command_string_stream.str();
		
		executeCommand(command.c_str(), file_name);

		// clear the contents of stringstream
/*		command_string_stream.str(std::string());
		command_string_stream << "echo Hello World!";
		std::cout << "Testing Compile command --> " << command_string_stream.str() << std::endl;
		*/
	}

	inline void executeCommand(	const std::string &command, 
								const char *file_name
	)
	{
		// std::cout << "Executing Compile command --> " << command << std::endl;
		error = system(command.c_str());
		// std::cout << "Compile Successful, file --> " << file_name << std::endl;

		if (error != 0)
		{
			std::cerr << "failed to compile " << file_name << std::endl;
			exit(error);
		}
	}
};

#endif //__CUDACOMPILE__H_