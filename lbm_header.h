/**
 * <<< prefix created from cpu program >>>
 *
 * the cpu program defines:
 *  - DOMAIN_CELLS
 *  - LOCAL_WORK_GROUP_SIZE
 *  - DOMAIN_CELLS_X
 *  - DOMAIN_CELLS_Y
 *  - DOMAIN_CELLS_Z
 *
 */

#include "lbm_defaults.h"

//#define USE_SWIZZLING	0

#define DOMAIN_CELLS		(DOMAIN_CELLS_X*DOMAIN_CELLS_Y)
#define DOMAIN_SLICE_CELLS	(DOMAIN_CELLS_X)
