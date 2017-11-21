#ifndef tsne_util_h
#define tsne_util_h

#include <cstdio>

void save(char *path, double* data, int n, int d) {
  FILE *h;
  if((h = fopen(path, "w")) == NULL) {
    printf("Error: could not open data file.\n");
    return;
  }
  
  for(int i = 0; i < n; ++i) {
    for(int j = 0; j < d; ++j)
      fprintf(h, "%lf ", data[i*d+j]);
    fprintf(h, "\n");
  }
  
  fclose(h);
  printf("Wrote the %i x %i data matrix successfully!\n", n, d);
}

#endif
