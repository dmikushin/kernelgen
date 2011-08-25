
hash_table *ident_hash;
struct line_maps line_table;

static int init_macros = 0;

static int source_line (cpp_reader * input, const struct line_map *map,
			source_location n);


/* linebuf allocator */

static int rank = 0;

static g95_linebuf *
get_linebuf (int size)
{
  g95_linebuf *k;
  int i;
  k =
    g95_getmem (sizeof (g95_linebuf) +
		(size + 1) * (sizeof (char) + sizeof (short)));
  k->rank = rank++;
  k->size = size;
  k->mask = (short *) ((char *) k->line + (size + 1) * sizeof (char));
  for (i = 0; i <= size + 1; i++)
    k->mask[i] = G95X_CHAR_NONE;
  return k;
}



/* cpp callbacks */

static const char *
missing_header (cpp_reader * input, const char *header, cpp_dir ** dir)
{
  FILE *fp = fopen (header, "r");
  if (fp == NULL)
    g95_fatal_error ("%s:%d: %s: %s",
		     current_name, current_file->line,
		     header, strerror (errno));
  fclose (fp);
  return header;
}



static void
define (cpp_reader * input, unsigned int k, cpp_hashnode * hnode)
{
  return;
#ifdef UNDEF
  const struct line_map *map =
    linemap_lookup (&line_table, input->out.first_line);
  int line = source_line (input, map, input->out.first_line);
  cpp_macro *macro = hnode->value.macro;

  printf ("<cpp_define id=\"0x%x\" macro=\"0x%x\"", hnode, macro);

  if (!init_macros)
    printf (" f=\"0x%x\" line=\"%d\"", current_file, line);

  g95x_end_obj ();

  g95x_print_macro (hnode);
#endif
}

static void
undef (cpp_reader * input, unsigned int k, cpp_hashnode * hnode)
{
  return;
#ifdef UNDEF
  const struct line_map *map =
    linemap_lookup (&line_table, input->out.first_line);
  int line = source_line (input, map, input->out.first_line);
  cpp_macro *macro = hnode->value.macro;

  if (!macro)
    return;

  void *m = hnode->flags & NODE_BUILTIN ? hnode : macro;

  printf ("<cpp_undef id=\"0x%x\" macro=\"0x%x\"", hnode, m);

  if (!init_macros)
    printf (" f=\"0x%x\" line=\"%d\"", current_file, line);

  g95x_end_obj ();
#endif
}


/* 
 * cpp helper functions; we have to manage memory allocation
 * for cpp, we do that by allocating ~ 4000b chunks of memory
 */


typedef struct _cpp_page_alloc
{
  struct _cpp_page_alloc *next;
  int length;
  int where;
  char mem[1];
} _cpp_page_alloc;

static _cpp_page_alloc *
_get_cpp_page_alloc (int length)
{
  _cpp_page_alloc *cpa =
    g95_getmem (sizeof (struct _cpp_page_alloc) + length);
  cpa->length = length;
  cpa->where = 0;
  return cpa;
}

static _cpp_page_alloc *_cpp_page_alloc_head = NULL,
  *_cpp_page_alloc_tail = NULL;

static void *
get_cpp_mem (int size)
{
  void *p = NULL;
  if (_cpp_page_alloc_head == NULL)
    {
      _cpp_page_alloc_head =
	_cpp_page_alloc_tail = _get_cpp_page_alloc (4000);
    }
  if (_cpp_page_alloc_tail->where + size >= _cpp_page_alloc_tail->length)
    {
      _cpp_page_alloc_tail->next = _get_cpp_page_alloc (4000 + size);
      _cpp_page_alloc_tail = _cpp_page_alloc_tail->next;
    }
  p = &_cpp_page_alloc_tail->mem[_cpp_page_alloc_tail->where];
  _cpp_page_alloc_tail->where += size;
  return p;
}

static void *
alloc_subobject (size_t x)
{
  return get_cpp_mem (x);
}

static hashnode
alloc_node (hash_table * table ATTRIBUTE_UNUSED)
{
  return get_cpp_mem (sizeof (struct cpp_hashnode));
}




static void
scanner_init ()
{
  static int done = 0;
  if (done == 0)
    {
      done++;
      ident_hash = ht_create (14);
      ident_hash->alloc_node = alloc_node;
      ident_hash->alloc_subobject = alloc_subobject;
    }
}

g95_linebuf *
g95x_get_line_head ()
{
  return line_head;
}

g95_linebuf *
g95x_get_line_tail ()
{
  return line_tail;
}

g95_file *
g95x_get_file_top ()
{
  g95_file *f;
  for (f = file_head; f; f = f->next)
    if (f->included_by == NULL)
      return f;
  return file_head;
}


g95_file *
g95x_get_file_head ()
{
  return file_head;
}

int
g95x_get_line_width ()
{
  return line_width;
}


int
g95x_current_line (g95_file * file)
{
  int line;
  cpp_reader *input;
  const struct line_map *map;

  input = file->x.input;
  map = linemap_lookup (&line_table, input->out.first_line);

  line = source_line (input, map, input->out.first_line);

  return line;
}

static void
g95x_set_linetot (g95_file* f)
{
  const char* pc;

  for (pc = &f->x.text[0]; *pc; pc++)
    if (*pc == '\n')
      f->x.linetot++;
 
}

static const char * get_tag (short int c, int close)
{
  switch (c)
    {
      case G95X_CHAR_NONE: return ""; break;
      case G95X_CHAR_CODE: return ""; break;
      case G95X_CHAR_COMM: return (!close) ?    "<C>" :    "</C>"; break;
      case G95X_CHAR_CTFX: return (!close) ? "<ctfx>" : "</ctfx>"; break;
      case G95X_CHAR_CTFR: return (!close) ? "<ctfr>" : "</ctfr>"; break;
      case G95X_CHAR_ZERO: return (!close) ? "<zero>" : "</zero>"; break;
      case G95X_CHAR_STRG: return (!close) ?    "<S>" :    "</S>"; break;
      case G95X_CHAR_SMCL: return (!close) ? "<smcl>" : "</smcl>"; break;
      case G95X_CHAR_RIMA: return ""; break;
      default:
        g95x_die("Unexpected char: `%c'", c);
    }
  return NULL;
}



static void g95x_dump_mask_simple (g95_linebuf *lb)
{
  int i, j;
  char c;
  for (i = 1; lb; lb = lb->next, i++)
    {
      printf ("%8d:%s\n", i, lb->line);
      printf ("%8d:", i);
      for (j = 0; lb->line[j]; j++)
        {
          switch (lb->mask[j])
            {
              case G95X_CHAR_NONE: c = ' '; break;
              case G95X_CHAR_CODE: c = ' '; break;
              case G95X_CHAR_COMM: c = '!'; break;
              case G95X_CHAR_CTFX: c = '>'; break;
              case G95X_CHAR_CTFR: c = '&'; break;
              case G95X_CHAR_ZERO: c = '0'; break;
              case G95X_CHAR_STRG: c = 'S'; break;
              case G95X_CHAR_SMCL: c = ';'; break;
              case G95X_CHAR_RIMA: c = 'x'; break;
              default:
                g95x_die ("");
            }
          printf ("%c", c);
        }
      printf ("\n");
    }
  
}

static void g95x_dump_mask_xml (g95_linebuf *lb)
{
  const char * linech;
  g95_file * f;
  int eol;
  int i, j, jlb;
  FILE * fp;

  fp = fopen ("mask.txt", "w");

  fprintf (fp, "\n");
  fprintf (fp, "<mask>\n");
  for (i = 1; lb; lb = lb->next, i++)
    {
      eol = 0;
      f = lb->file;
      linech = lb->line;

      for (j = 0; linech[j] != '\0'; j++)
        {
          jlb = j;
          if (!eol)
            {
              eol = eol || (lb->line[jlb] == '\0');
              if (eol && strcmp (get_tag (lb->mask[jlb-1], 1), ""))
                fprintf (fp, "%s", get_tag (lb->mask[jlb-1], 1));
            }
          if (eol)
            {
               fprintf (fp, "%c", linech[j]);
            }
          else
            {
              if (j == 0)
                fprintf (fp, "%s%c", get_tag (lb->mask[jlb], 0), linech[j]);
              else if (lb->mask[jlb-1] != lb->mask[jlb])
                fprintf (fp, "%s%s%c", 
                                  get_tag (lb->mask[jlb-1], 1),
                                  get_tag (lb->mask[jlb], 0), linech[j]);
              else
                fprintf (fp, "%c", linech[j]);
            }
        }
      if ((j > 0) && (!eol))
        {
          jlb = j-1;
          if (strcmp (get_tag (lb->mask[jlb], 1), ""))
            fprintf (fp, "%s", get_tag (lb->mask[jlb], 1));
        }
      fprintf (fp, "\n");
    }
  fprintf (fp, "</mask>\n");
  fclose (fp);
}


static void g95x_dump_mask (g95_linebuf *lb)
{
  g95x_dump_mask_xml (lb);
}
