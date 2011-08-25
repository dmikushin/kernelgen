#ifndef _XML_CPP_H
#define _XML_CPP_H

typedef struct g95x_cpp_item
{
  struct g95x_cpp *cpp;
  short int c1, c2;
} g95x_cpp_item;

typedef struct g95x_cpp
{
  struct g95x_cpp *next;
  void *macro;
  short int c1, c2;
  short int n;
  g95x_cpp_item s[0];
} g95x_cpp;

typedef struct g95x_cpp_line
{
  int line;
  g95x_cpp *xcp;
  struct g95x_cpp_line *next;
} g95x_cpp_line;

#define g95x_get_cpp_line() (g95x_cpp_line*)g95_getmem( sizeof( g95x_cpp_line ) )
#define g95x_free_cpp_line( xcl ) g95_free( xcl )

g95x_cpp *g95x_get_cpp (int ss, int st);

void g95x_push_cpp (g95x_cpp * xcp);

struct g95_file;
struct g95_linebuf;
void g95x_push_cpp_line (struct g95_file *, struct g95_linebuf *);
struct g95x_cpp_mask *g95x_delta_back (struct g95_linebuf *);


#endif
