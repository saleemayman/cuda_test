//#ifndef __DEFAULTS__H_
//#define __DEFAULTS__H_

#if DEFAULTS
	#define DOMAIN_CELLS_X	(1)
	#define DOMAIN_CELLS_Y	(3)
	#define LOCAL_WORK_GROUP_SIZE (32)

//	#define LOCAL_WORK_GROUP_SIZE	(128)

#endif	//DEFAULTS

#ifndef SIMULATION_DATA_TYPE
#define SIMULATION_DATA_TYPE
	#if TYPE_FLOAT
		typedef float	T;
	#elif TYPE_DOUBLE
		typedef double	T;
	#endif
#endif //SIMULATION_DATA_TYPE


// #if DEFAULTS
// 	#define DOMAIN_CELLS_X	(1)
// 	#define DOMAIN_CELLS_Y	(2)
// #endif	//DEFAULTS

// #ifndef SIMULATION_DATA_TYPE
// #define SIMULATION_DATA_TYPE
// 	#if TYPE_FLOAT
// 		typedef float	T;
// 		#endif
// 	#elif TYPE_DOUBLE
// 		typedef double	T;
// 	#endif
// #endif	//SIMULATION_DATA_TYPE