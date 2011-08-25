typedef struct g95x_locus
{
  struct g95_linebuf *lb1;
  int column1;
  struct g95_linebuf *lb2;
  int column2;
} g95x_locus;

#ifdef G95X_64BITS
typedef unsigned long long g95x_pointer;
#else
typedef unsigned int g95x_pointer;
#endif
