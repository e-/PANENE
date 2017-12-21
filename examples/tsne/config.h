#define USE_ADAM 0 // ADAM or the original algorithm?

#define USE_EE 1 // use early exaggeration (ee)?
#define PERIODIC_EE 1 // periodic EE (e.g., 30 iterations for every 100 iterations)
#define PERIODIC_RESET 0 // when applying ee, reset the momentum of the optimizer
#define EE_FACTOR 12.0f 


/*
 *
 * momentum vs adam
 * use EE or not
 */
