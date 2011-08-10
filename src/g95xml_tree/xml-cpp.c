#include "g95.h"
#include "xml-cpp.h"

#include "config.h"
#include "system.h"
#include "cpplib.h"
#include "internal.h"

#include <assert.h>



g95x_cpp *
g95x_get_cpp (int ss, int st)
{
  int i;
  g95x_cpp *xcp;
  xcp =
    (g95x_cpp *) g95_getmem (sizeof (g95x_cpp) +
			     ss * 2 * sizeof (g95x_cpp_item) + st);
  for (i = 0; i < ss; i++)
    xcp->s[i].cpp = xcp;
  xcp->n = ss;
  xcp->next = NULL;
  return xcp;
}

static g95x_cpp *xcp_list = NULL;

void
g95x_push_cpp (g95x_cpp * xcp)
{
  xcp->next = xcp_list;
  xcp_list = xcp;
}

void
g95x_push_cpp_line (struct g95_file *current, struct g95_linebuf *lb)
{
  g95x_cpp_line *xcl;

  if (xcp_list == NULL)
    return;
  xcl = g95x_get_cpp_line ();
  xcl->next = current->x.cl;
  current->x.cl = xcl;

  lb->xcp = xcp_list;

  xcl->xcp = xcp_list;
  xcl->line = g95x_current_line (current);
  xcp_list = NULL;
}

#ifdef UNDEF
static int
_loc1 (g95x_cpp * xcp, int c)
{
  int cl0 = xcp->c1;
  int i;

  if (c <= cl0)
    return c;

  for (i = 0; i < xcp->n; i++)
    {
      g95x_cpp_item *xci = &xcp->s[i];
      int cst = xci->c1 < 0;
      char *txt = cst ? (char *) xci->cpp + xci->c2 : NULL;
      int cl1 = cl0 + (cst ? strlen (txt) : xci->c2 - xci->c1);

      if (c <= cl1)
	{
	  if (cst)
	    return -1;
	  else
	    return c - cl0 + xci->c1;
	}
      cl0 = cl1;
    }
  return c - cl0 + xcp->c2;
}

static int
cpp_back (g95_linebuf * lb, int c)
{
  g95x_cpp *xcp;
  printf (" %2d", c);
  for (xcp = lb->xcp; xcp; xcp = xcp->next)
    {
      c = _loc1 (xcp, c);
      if (c < 0)
	break;
      printf (" > %2d", c);
    }
  if (c < 0)
    printf (" >  X");
  printf ("\n");
  return c;
}

#endif

g95x_cpp_mask *
g95x_get_cpp_mask (int norig, int ncode)
{
  g95x_cpp_mask *cpp_mask =
    (g95x_cpp_mask *) g95_getmem (sizeof (g95x_cpp_mask) +
				  sizeof (short) * norig +
				  sizeof (g95x_cpp_mask_item) * ncode);
  cpp_mask->norig = norig;
  cpp_mask->ncode = ncode;
  cpp_mask->orig = (short *) ((char *) cpp_mask + sizeof (g95x_cpp_mask));
  cpp_mask->code =
    (g95x_cpp_mask_item *) ((char *) cpp_mask->orig + sizeof (short) * norig);
  return cpp_mask;
}

static int
_loc1 (g95x_cpp * xcp, int c)
{
  int cl0 = xcp->c1;
  int i;

  if (c <= cl0)
    return c;

  for (i = 0; i < xcp->n; i++)
    {
      g95x_cpp_item *xci = &xcp->s[i];
      int cst = xci->c1 < 0;
      char *txt = cst ? (char *) xci->cpp + xci->c2 : NULL;
      int cl1 = cl0 + (cst ? strlen (txt) : xci->c2 - xci->c1);

      if (c <= cl1)
	return -1;
      cl0 = cl1;
    }
  return c - cl0 + xcp->c2;
}

static int
delta (g95x_cpp * xcp)
{
  int len = 0;
  int i;
  for (i = 0; i < xcp->n; i++)
    {
      g95x_cpp_item *xci = &xcp->s[i];
      int cst = xci->c1 < 0;
      char *txt = cst ? (char *) xci->cpp + xci->c2 : NULL;
      len += (cst ? strlen (txt) : xci->c2 - xci->c1);
    }
  return xcp->c2 - xcp->c1 - len;
}

g95x_cpp_mask *
g95x_delta_back (g95_linebuf * lb)
{
  g95x_cpp *xcp;
  short *p0, *p1;
  int i, j, k = 0, i0, i1;
  int len0, len1;
  g95x_cpp_mask *cpp_mask;
  int llb = strlen (lb->line);

  if (lb->xcp == NULL)
    return NULL;

#define PA( size ) (short*)g95_getmem( sizeof( short ) * ( size ) )
#define pP do { for( i = 0; i < len0; i++ ) printf( " %2.2d", p0[i] ); printf( "\n" ); } while(0)
#undef DELTADEBUG

  len0 = llb;
  p0 = PA (len0);
  for (i = 0; i < len0; i++)
    p0[i] = i + 1;

  for (xcp = lb->xcp; xcp; xcp = xcp->next, k++, len0 = len1, p0 = p1)
    {
#ifdef DELTADEBUG
      pP;
#endif
      len1 = len0 + delta (xcp);
#ifdef DELTADEBUG
      printf (" len1 = %d, len0 = %d, c1 = %d, c2 = %d\n", len1, len0,
	      xcp->c1, xcp->c2);
#endif
      p1 = PA (len1);
      for (i0 = 0, i1 = 0; i0 < len0; i0++, i1++)
	{
#ifdef DELTADEBUG
	  printf (" len1 = %d, i1 = %d, len0 = %d, i0 = %d, P0 = %d\n", len1,
		  i1, len0, i0, p0[i0]);
#endif
	  if (i1 + 1 < xcp->c1)
	    {
	      p1[i1] = p0[i0];
	    }
	  else if (i1 + 1 < xcp->c2)
	    {
	      int k = xcp->c1 > 0 ? 0 : 1;
	      if (k == 0)
		p1[i1] = p0[i0];
#ifdef DELTADEBUG
	      printf ("---\n");
#endif
	      i0 += (xcp->c2 - xcp->c1) - (len1 - len0) - k;
	      i1 += xcp->c2 - xcp->c1 - k;
	    }
	  else
	    {
#ifdef DELTADEBUG
	      printf ("+++\n");
#endif
	      if (!(i1 < len1))
		g95x_die ("");
	      p1[i1] = p0[i0];
	    }
	}
      g95_free (p0);
    }
#ifdef DELTADEBUG
  pP;
#endif


  g95x_pointer id = 0;
  int kc1 = 0, kc2 = 0;
  cpp_mask = g95x_get_cpp_mask (len0, llb);
  for (i = 0; i < cpp_mask->norig; i++)
    {
      cpp_mask->orig[i] = p0[i];
      if (p0[i] > 0)
	{
	  for (j = 0; j < p0[i] - 1; j++)
	    if ((cpp_mask->code[j].c == 0) && (cpp_mask->code[j].c1 == 0))
	      {
		if (id == 0)
		  id = g95x_get_obj_id_an ();
		cpp_mask->code[j].c1 = kc1;
		cpp_mask->code[j].c2 = kc2;
		cpp_mask->code[j].id = id;
	      }
	    else
	      {
		id = 0;
	      }
	  kc1 = kc2 = 0;
	  cpp_mask->code[p0[i] - 1].c = i + 1;
	}
      else if (kc1 == 0)
	{
	  kc2 = kc1 = i + 1;
	}
      else
	{
	  kc2 = i + 1;
	}
    }
  for (j = 0; j < llb; j++)
    if ((cpp_mask->code[j].c == 0) && (cpp_mask->code[j].c1 == 0))
      {
	if (id == 0)
	  id = g95x_get_obj_id_an ();
	cpp_mask->code[j].c1 = kc1;
	cpp_mask->code[j].c2 = kc2;
	cpp_mask->code[j].id = id;
      }
    else
      {
	id = 0;
      }

  g95_free (p0);
  return cpp_mask;
}


static int
cpp_back (g95_linebuf * lb, int c)
{
  g95x_cpp *xcp;
  printf (" %2d", c);
  for (xcp = lb->xcp; xcp; xcp = xcp->next)
    {
      c = _loc1 (xcp, c);
      if (c < 0)
	break;
      printf (" > %2d", c);
    }
  if (c < 0)
    printf (" >  X");
  printf ("\n");
  return c;
}

int
g95x_cpp_back (g95_linebuf * lb)
{
  int c;
  for (; lb; lb = lb->next)
    {
      printf (" %d \n", lb->linenum);
      for (c = 1; c <= lb->size; c++)
	cpp_back (lb, c);
      printf ("\n");
      g95x_delta_back (lb);
    }
  return 0;
}
