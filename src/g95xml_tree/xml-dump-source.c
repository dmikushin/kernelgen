#include "g95.h"
#include "xml-cpp.h"
#include "xml-string.h"


#define G95X_QUOT "&quot;"
#define G95X_LT   "&lt;"
#define G95X_GT   "&gt;"
#define G95X_AMP  "&amp;"
#define G95X_ESC  "&#%3.3d;"

#define G95X_LEN_QUOT 6
#define G95X_LEN_LT   4
#define G95X_LEN_GT   4
#define G95X_LEN_AMP  5
#define G95X_LEN_ESC  7


void
g95x_xml_escape (char *d, const char *s, int n)
{
  int i, j;
  unsigned char u;
  for (i = 0, j = 0; s[j] && (j < n); j++)
    {
      switch (s[j])
	{
	case '"':
	  strcpy (&d[i], G95X_QUOT);
	  i += G95X_LEN_QUOT;
	  break;
	case '<':
	  strcpy (&d[i], G95X_LT);
	  i += G95X_LEN_LT;
	  break;
	case '>':
	  strcpy (&d[i], G95X_GT);
	  i += G95X_LEN_GT;
	  break;
	case '&':
	  strcpy (&d[i], G95X_AMP);
	  i += G95X_LEN_AMP;
	  break;
	default:
	  u = (unsigned char) s[j];
	  if ((u > 31) && (u < 127))
	    {
	      d[i] = s[j];
	      i++;
	    }
	  else if ((u >= 127) || (u == 9) || (u == 10) || (u == 13))
	    {
	      sprintf (&d[i], G95X_ESC, (unsigned char) s[j]);
	      i += G95X_LEN_ESC;
	    }
	  else
	    {
	      g95x_die ("");
	      d[i] = '?';
	      i++;
	    }
	}
    }
  d[i] = '\0';
}

static void
dump_cpp (const char *line, int len, g95x_pointer id, int *open)
{
  static char temp[30000];
  g95x_xml_escape (temp, line, len);
  g95x_put_txt1 ("cpp-text", temp, 0, id, open);
}

static char
dump_txt (const char *line, const short *mask, int len,	/* text to dump, mask, length                    */
	  int cpp, g95x_pointer id, int *open	        /* in cpp affected, id of cpp segment, open flag */
  )
{
  static char temp[30000];
  int i1, i, i2;
  const char *class;
  int m1 = 0;

  for (i1 = 0; i1 < len;)
    {
      m1 = mask[i1];
      for (i = i1; i < len; i++)
	{
	  if ((i > i1) && (m1 & (G95X_CHAR_CTFX | G95X_CHAR_CTFR
				 | G95X_CHAR_ZERO | G95X_CHAR_SMCL)))
	    break;
	  if (mask[i] == G95X_CHAR_NONE)
	    continue;
	  if ((i > i1) && (mask[i] & (G95X_CHAR_CTFX | G95X_CHAR_CTFR
				      | G95X_CHAR_ZERO | G95X_CHAR_SMCL)))
	    break;
	  if ((m1 == G95X_CHAR_NONE) && (mask[i] != G95X_CHAR_COMM))
	    {
	      m1 = mask[i];
	    }
	  else if (m1 != mask[i])
	    {
	      break;
	    }
	}
      i2 = i;
      g95x_xml_escape (temp, &line[i1], i2 - i1);
      switch (m1)
	{
	case G95X_CHAR_NONE:
	  class = G95X_S_SPACE;
	  break;
	case G95X_CHAR_CODE:
	  class = G95X_S_CODE;
	  break;
	case G95X_CHAR_COMM:
          if (g95x_option.xml_example)
            g95x_xml_example_switch (temp);
	  class = G95X_S_COMMENT;
	  break;
	case G95X_CHAR_SMCL:
	  class = G95X_S_SEMI G95X_S_HN G95X_S_COLUMN;
	  break;
	case G95X_CHAR_CTFX:
	  class = G95X_S_CONT G95X_S_HN G95X_S_FIXED;
	  break;
	case G95X_CHAR_CTFR:
	  class = G95X_S_CONT G95X_S_HN G95X_S_FREE;
	  break;
	case G95X_CHAR_ZERO:
	  class = G95X_S_ZERO;
	  break;
	case G95X_CHAR_STRG:
	  class = G95X_S_STRING;
	  break;
	case G95X_CHAR_RIMA:
	  class = G95X_S_RIGHT G95X_S_HN G95X_S_MARGIN;
	  break;
	default:
	  g95x_die ("");
	}
      g95x_put_txt1 (class, temp, cpp, id, open);
      i1 = i2;
    }
  return m1;
}

static void
f_inc_linenum (g95_file * f)
{
  if (f->x.linenum > f->x.linetot)
    return;
  f->x.linenum++;
  while (1)
    {
      f->x.cur++;
      if (f->x.cur[-1] == '\n')
	break;
    }
//printf ("%s:%d\n", f->filename, f->x.linenum);
}


static char uns_temp[30000];
static void
dump_uns (g95_file * f, int linenum, int *open)
{
  const char *str;
  while (f->x.linenum < linenum)
    {
      str = f->x.cur;
      f_inc_linenum (f);
      g95x_xml_escape (uns_temp, str, f->x.cur - str - 1);
      if (uns_temp[0])
        g95x_put_txt1 (str[0] == '#' ? G95X_S_CPP : G95X_S_UNSEEN, 
                       uns_temp, 0, 0, open);
      if (f->x.cur[0])
	g95x_put_txt1 (G95X_S_BR, NULL, 0, 0, open);
    }
}

#define isspc( c ) ( ( (c) ==  ' ' ) || ( (c) == '\t' ) )

static void
dump_include_line (g95_file * f, int *open)
{
  int n1, n2, n3, n4;
  const char *cls;
  char q;
  if (f->x.cur[0] == '#')
    {
      cls = G95X_S_CPP G95X_S_HN G95X_S_INCLUDE;
      n1 = 0;
      if (strncmp (&f->x.cur[1], "include", 7))
	g95x_die ("");
      n2 = n1 + 8;
    }
  else
    {
      cls = G95X_S_FORTRAN G95X_S_HN G95X_S_INCLUDE;
      for (n1 = 0; isspc (f->x.cur[n1]); n1++);
      if ((n1 > 0) && (!isspc (f->x.cur[n1 - 1])))
	g95x_die ("");
      if (strncasecmp (&f->x.cur[n1], "include", 7))
	g95x_die ("");
      n2 = n1 + 7;
    }

  for (; isspc (f->x.cur[n2]); n2++);
  q = f->x.cur[n2++];

  for (n3 = n2 + 1; f->x.cur[n3] && (f->x.cur[n3] != q); n3++);
  if (f->x.cur[n3] == '\0')
    g95x_die ("");

  n4 = index (&f->x.cur[0], '\n') - &f->x.cur[0] - 1;

  if (n1 > 0)
    {
      g95x_xml_escape (uns_temp, &f->x.cur[0], n1);
      g95x_put_txt1 (G95X_S_SPACE, uns_temp, 0, 0, open);
    }

  g95x_add_sct1 (cls);
  g95x_xml_escape (uns_temp, &f->x.cur[n1], n2 - n1);
  g95x_put_txt1 (G95X_S_CODE, uns_temp, 0, 0, open);

  g95x_xml_escape (uns_temp, &f->x.cur[n2], n3 - n2);
  g95x_put_txt1 (G95X_S_FILENAME, uns_temp, 0, 0, open);

  g95x_xml_escape (uns_temp, &f->x.cur[n3], 1);
  g95x_put_txt1 (G95X_S_CODE, uns_temp, 0, 0, open);
  g95x_end_sct1 ();

  if (n4 - n3 - 1 > 0)
    {
      g95x_xml_escape (uns_temp, &f->x.cur[n3 + 1], n4 - n3 - 1);
      g95x_put_txt1 (G95X_S_UNSEEN, uns_temp, 0, 0, open);
    }


  f_inc_linenum (f);

  g95x_end_txt1 (open);
}


static void
enter_file (g95_file * fo, g95_file * fn, int *open)
{
  dump_uns (fo, fn->inclusion_line, open);
  g95x_end_txt1 (open);
  if (fn->up != fo)
    enter_file (fo, fn->up, open);
  if (fo)
    g95x_add_inc ();
  g95x_add_sct1 (G95X_S_INCLUDED G95X_S_HN G95X_S_FILE);
  g95x_add_fle (fn);
}

/* leave a file, dumping tail of text as unseen
 * and quiting intermediate files, up to fn
 */
static void
quit_file (g95_file * fo, g95_file * fn, int *open)
{
  g95x_end_txt1 (open);
  while (fo->x.linenum < fo->x.linetot)
    {
      g95x_put_txt1 (G95X_S_BR, NULL, 0, 0, open);
      dump_uns (fo, fo->x.linenum + 1, open);
    }
  g95x_end_txt1 (open);
  g95x_end_fle ();
  if (fo->up)
    {
      g95x_end_sct1 ();
      dump_include_line (fo->up, open);
      g95x_end_inc ();
    }
  if (fo->up != fn)
    quit_file (fo->up, fn, open);
}

/* look for the first common ancestor of two files
 */
static g95_file *
fancestor (g95_file * f1, g95_file * f2)
{
  while (f1->x.include_level > f2->x.include_level)
    f1 = f1->up;
  while (f2->x.include_level > f1->x.include_level)
    f2 = f2->up;
  while (f1 != f2)
    {
      f1 = f1->up;
      f2 = f2->up;
    }
  return f1;
}

/* manage file change, going through intermediate files, if any
 * fo is the old file object, fn is the new file 
 * we have to look for the first file ancestor, 
 * and leave the chain fo -> fa
 * then enter fa -> fn
 */
static void
change_file (g95_file * fo, g95_file * fn, int *open)
{
  g95_file *fa = fancestor (fo, fn);
  if (fo != fa)
    {
      quit_file (fo, fa, open);
      if (fn == fa)
	g95x_put_txt1 (G95X_S_BR, NULL, 0, 0, open);
    }
  if (fa != fn)
    enter_file (fa, fn, open);
}

static g95_linebuf *current_lb = NULL;
static int current_column = 0;
static g95x_cpp_mask *cpp_mask = NULL;


static void
dump_cpp_line (int len, int *open)
{
  int dump_cpp_now = 0,		/* Do we dump cpp-text now                 */
    dump_cpp_pos,		/* First char of cpp-text                  */
    dump_cpp_len,		/* Length of cpp-text                      */
    dump_cpp_k = -1,		/* Offset of the cpp-text in the code      */
    cpp = 0;			/* Number of chars of cpp-text in the code */
  g95x_pointer dump_cpp_id = 0;	/* id of the cpp segment we are in */

  int k;
  for (k = 0; k < len + 1; k++)
    {
      if ((k < len) && (cpp_mask->code[current_column + k - 1].c == 0))
	{
	  dump_cpp_k = dump_cpp_k < 0 ? k : dump_cpp_k;
	  cpp++;
	  if (cpp)
	    dump_cpp_id = cpp_mask->code[current_column + k - 1].id;
	}
      if ((current_column + k > 1)
	  && (cpp_mask->code[current_column + k - 2].c == 0)
	  && ((current_column + k > current_lb->size)
	      || (cpp_mask->code[current_column + k - 1].c > 0)))
	{

	  int j;		/* Search back the first element with c == 0 */

	  for (j = 2; current_column + k - j >= 0; j++)
	    {
	      if (cpp_mask->code[current_column + k - j].c != 0)
		{
		  j--;
		  break;
		}
	    }
	  if (current_column + k - j < 0)
	    j = current_column + k;


	  dump_cpp_now = !cpp_mask->code[current_column + k - j].done++;
	  dump_cpp_pos = cpp_mask->code[current_column + k - j].c1 - 1;
	  dump_cpp_len = cpp_mask->code[current_column + k - j].c2 -
	    cpp_mask->code[current_column + k - j].c1 + 1;
	}
    }

  if (dump_cpp_now)
    {
      int offset = dump_cpp_k < 0 ? 0 : dump_cpp_k;
      if (offset > 0)
	dump_txt (&current_lb->line[current_column - 1],
		  &current_lb->mask[current_column - 1],
		  offset, 0, dump_cpp_id, open);

      dump_txt (&current_lb->line[current_column - 1 + offset],
		&current_lb->mask[current_column - 1 + offset],
		cpp, 1, dump_cpp_id, open);

      dump_cpp (current_lb->file->x.cur + dump_cpp_pos, dump_cpp_len,
		dump_cpp_id, open);

      dump_txt (&current_lb->line[current_column - 1 + cpp + offset],
		&current_lb->mask[current_column - 1 + cpp + offset],
		len - cpp - offset, 0, dump_cpp_id, open);

    }
  else
    {

      dump_txt (&current_lb->line[current_column - 1],
		&current_lb->mask[current_column - 1],
		len, cpp, dump_cpp_id, open);

    }
}

int g95x_dump_source_enabled = 1;

void
g95x_dump_source_code (g95x_locus * xw, int k, int r)
{
  if (!g95x_dump_source_enabled) return;

  static int first = 1;
  g95_linebuf *end_lb;
  int end_column;
  int len;
  int open = 0;


  if (first)
    {
      current_lb = g95x_get_line_head ();
      current_column = 1;
      first = 0;
      cpp_mask = g95x_delta_back (current_lb);
    }
  else if (current_lb == NULL)
    {
      return;
    }

  if (xw == NULL)
    g95x_die ("");

  if (xw->lb1 == NULL)
    {
      if (k == 0)
        end_lb = NULL;
      else
        g95x_die ("");
    }
  else if (xw->lb2 == NULL)
    {
      if (k != 0)
        end_lb = NULL;
      else
        g95x_die ("");
    }
  else if (r)
    {
      if (k == 0)
	g95x_refine_location1 (xw, &end_lb, &end_column);
      else
	g95x_refine_location2 (xw, &end_lb, &end_column);
    }
  else
    {
      if (k == 0)
	{
	  end_lb = xw->lb1;
	  end_column = xw->column1;
	}
      else
	{
	  end_lb = xw->lb2;
	  end_column = xw->column2;
	}
    }


  if (current_lb->file->x.linenum < current_lb->linenum)
    dump_uns (current_lb->file, current_lb->linenum, &open);

  while (1)
    {
      if (end_lb == current_lb)
	len = end_column - current_column;
      else
	len = current_lb->size - current_column + 1;
      if (len < 0)
	g95x_die ("g95x_dump_source_code: %d %d %d",
		  current_lb->linenum, current_column, end_column);

      if (cpp_mask)
	dump_cpp_line (len, &open);
      else
	dump_txt (&current_lb->line[current_column - 1],
		  &current_lb->mask[current_column - 1], len, 0, 0, &open);

      current_column += len;

      if (end_lb == current_lb)
	break;

      f_inc_linenum (current_lb->file);

      current_column = 1;

      g95_linebuf *next_lb = current_lb->next;


      if (cpp_mask)
	g95_free (cpp_mask);
      if (next_lb == NULL)
	break;

      if (current_lb->file == next_lb->file)
	{
	  g95x_put_txt1 (G95X_S_BR, NULL, 0, 0, &open);
	}
      else
	{
	  g95x_end_txt1 (&open);
	  change_file (current_lb->file, next_lb->file, &open);
	}

      current_lb = next_lb;

      cpp_mask = g95x_delta_back (current_lb);


      if (current_lb->file->x.linenum < current_lb->linenum)
	dump_uns (current_lb->file, current_lb->linenum, &open);

    }
  g95x_end_txt1 (&open);
}

int
g95x_close_file (g95_file * f)
{
  g95x_locus xw;
  g95_file *up = f->up;
  g95_linebuf *lb;
  int open = 0;

  if (f->x.closed)
    g95x_die ("");

//printf ("%s\n", f->filename);

  if (up == NULL)
    {
      xw.lb1     = xw.lb2     = NULL;
      xw.column1 = xw.column2 = 0;
      g95x_dump_source_code (&xw, 0, 0);
      quit_file (f, NULL, &open);
      g95x_end_txt1 (&open);
      goto end;
    }


  for (lb = current_lb; lb; lb = lb->next)
    if ((lb->file == up) && (lb->linenum > f->inclusion_line))
      break;

  if (lb != NULL)
    {
      xw.lb1     = xw.lb2     = lb;
      xw.column1 = xw.column2 = 1;
      g95x_dump_source_code (&xw, 0, 0);
    }
  else
    {
      /*
       * Special case; include at EOF
       */
      if (current_lb)
        {
          xw.lb1     = xw.lb2     = current_lb->next;
          xw.column1 = xw.column2 = 1;
          g95x_dump_source_code (&xw, 0, 0);
          for (lb = current_lb->next; lb; lb = lb->next)
            {
              xw.lb1     = xw.lb2     = lb;
              xw.column1 = xw.column2 = 1;
              g95x_dump_source_code (&xw, 0, 0);
            }
          current_lb = NULL;
        }
      quit_file (f, up, &open);

    }

end:
  f->x.closed++;
  return 0;
}
