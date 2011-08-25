#include "g95.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "xml-string.h"

#undef G95X_ADD_ADDR
#undef G95X_INDENT
#define G95X_CR ""
#define G95X_BR "<%s" G95X_S_L "/>"
#define G95X_CRBR "<%s" G95X_S_L ">\n</%s" G95X_S_L ">"


static char *prefix;

static int pstack = -1;
FILE *g95x_out = NULL;
static char xml_f[256];
static char * g95x_f;


char *
g95x_f2xml (const char *file)
{
  int length;
  int i;

  length = strlen (file);
  strcpy (xml_f, file);

  for (i = length; i >= 0; i--)
    if (xml_f[i] == '.')
      break;
  if (i == 0)
    i = length - 1;


  xml_f[i + 1] = 'o';
  xml_f[i + 2] = '\0';

  return xml_f;
}

#ifdef G95X_INDENT
static void
indent (int k)
{
  int i;
  for (i = 0; i < pstack + k; i++)
    fprintf (g95x_out, "  ");
}
#else
#define indent( k ) do{}while(0)
#endif


int
g95x_init_options (int argc, char *argv[])
{
  int i;

  g95x_f = argv[argc - 1];

  for (i = 0; i < argc; i++)
  {
    if ((!strcmp(argv[i], "-o")) && (i < argc-1))
      g95x_f = argv[i+1];
    if (!strcmp(argv[i], "-xml-out=-"))
    {
      g95x_out = stdout;
      return 1;
    }
  }

  g95x_out = fopen (g95x_f2xml (g95x_f), "w");
  if (! g95x_out)
    g95x_die ("Cannot open output file\n");
  return 1;
}

const char*
g95x_get_f95_file ()
{
  return g95x_f;
}

const char*
g95x_get_xml_file ()
{
  return xml_f;
}


g95x_pointer
g95x_get_obj_id_an ()
{
  g95x_bbt_id *st = g95x_new_bbt_id_an (&root_bbt_id);
  return st->q;
}

static g95x_pointer
get_obj_id (int def, g95x_pointer p, int ni)
{
  g95x_bbt_id *st = g95x_get_bbt_id (p, NULL, &root_bbt_id);
  if (!st)
    st = g95x_new_bbt_id (p, NULL, &root_bbt_id);
  if (def > 0)
    {
      st->defined++;
      st->hidden = ni ? 1 : 0;
    }
  else
    {
      st->referenced = 1;
    }
  return st->q;
}

static g95x_pointer
get_als_id (int def, const char *als, g95x_pointer p)
{
  g95x_bbt_id *st = g95x_get_bbt_id (p, als, &root_bbt_id);
  if (!st)
    st = g95x_new_bbt_id (p, als, &root_bbt_id);
  if (def > 0)
    st->defined++;
  return st->q;
}

int
g95x_exists_obj_id (g95x_pointer p)
{
  return g95x_get_bbt_id (p, NULL, &root_bbt_id) != NULL;
}

int
g95x_exists_als_id (const char *als, g95x_pointer p)
{
  return g95x_get_bbt_id (p, als, &root_bbt_id) != NULL;
}

int
g95x_defined_obj_id (g95x_pointer p)
{
  g95x_bbt_id *st;
  if ((st = g95x_get_bbt_id (p, NULL, &root_bbt_id)))
    if (st->defined)
      return 1;
  return 0;
}

int
g95x_defined_als_id (const char *als, g95x_pointer p)
{
  g95x_bbt_id *st;
  if ((st = g95x_get_bbt_id (p, als, &root_bbt_id)))
    if (st->defined)
      return 1;
  return 0;
}


#define MAX_OBJ 1000
static struct
{
  enum
  {
    G95X_STACK_OBJ = 0, G95X_STACK_LST = 1,
    G95X_STACK_SCT = 2, G95X_STACK_FLE = 3,
    G95X_STACK_INC = 4
  } type;
  int nelems;
  const char *name;
  const char *T;
  int end;
  g95_file *f;
  int done;
} stack[MAX_OBJ];

#define g95x_clear_stack(istack) \
  do { memset (&stack[istack], 0, sizeof(stack[istack])); } while (0)

#define g95x_r_stack() \
  do {                          \
    if (pstack > MAX_OBJ)       \
      g95x_die ("");            \
    pstack++;                   \
    g95x_clear_stack (pstack);  \
  } while (0)

#define g95x_l_stack() \
  do {                          \
    if (pstack < 0)             \
      g95x_die ("");            \
    pstack--;                   \
  } while (0)

#define g95x_c_stack(k) (stack[pstack+k])
#define g95x_c_0 g95x_c_stack(0)
#define g95x_c_1 g95x_c_stack(-1)


const char* g95x_current_obj_name ()
{
  return pstack >= 0 ? g95x_c_0.name : NULL;
}

const char* g95x_current_obj_type ()
{
  return pstack >= 0 ? g95x_c_0.T : NULL;
}



int
g95x_add_dmp ()
{
  if (g95x_option.xmlns)
    {
      int len = strlen (g95x_option.xmlns);
      prefix = (char *) malloc (len + 2);
      strcpy (prefix, g95x_option.xmlns);
      strcat (prefix, ":");
    }
  else
    {
      static char c = '\0';
      prefix = &c;
    }
  if (!g95x_option.xml_no_header)
    fprintf (g95x_out, "<?xml version=\"1.0\"?>");
  if (g95x_option.stylesheet)
    fprintf (g95x_out, "<?xml-stylesheet type=\"%s\" href=\"%s\"?>",
	     g95x_option.stylesheet_type ? g95x_option.
	     stylesheet_type : "text/css", g95x_option.stylesheet);
  if (g95x_option.xml_xul) 
    fprintf (g95x_out, "<window flex=\"1\" xmlns=\"http://www.mozilla.org/keymaster/gatekeeper/there.is.only.xul\"><scrollbox flex=\"1\" style=\"overflow:auto;\">");

  if (!g95x_option.xml_no_ns)
    {
      fprintf (g95x_out, "<%s" G95X_S_FORTRAN95 " xmlns", prefix);
      if (g95x_option.xmlns)
        fprintf (g95x_out, ":%s", g95x_option.xmlns);
      fprintf (g95x_out, "=\"%s\">",
	   g95x_option.xmlns_uri ? g95x_option.
	   xmlns_uri : "http://g95-xml.sourceforge.net/");
    }
  else
    {
      fprintf (g95x_out, "<" G95X_S_FORTRAN95 ">");
    }

  if (g95x_option.xml_watch)
    g95x_add_watch ();

  return 1;
}

int
g95x_end_dmp ()
{
  fprintf (g95x_out, "</" G95X_S_FORTRAN95 ">");
  if (g95x_option.xml_xul) 
    fprintf (g95x_out, "</scrollbox></window>");
  fclose (g95x_out);
  g95x_out = NULL;
  if (g95x_option.xml_watch)
    g95x_end_watch ();
  return 1;
}

static int
add_lst1 ()
{
  if (pstack < 0)
    return 0;
  if (g95x_c_0.type != G95X_STACK_LST)
    return 0;
  if (g95x_c_0.nelems == 0)
    {
      if (pstack > 0)
	{
	  if (g95x_c_1.nelems == 0)
	    {
	      if (g95x_c_1.end++ == 0)
		fprintf (g95x_out, ">" G95X_CR);
	    }
	  g95x_c_1.nelems++;
	}
      indent (0);
      if (g95x_c_0.done++ == 0)
	return fprintf (g95x_out, "<%s%s>" G95X_CR, prefix, g95x_c_0.name);
    }

  g95x_c_0.nelems++;

  return 0;
}


int
g95x_add_als1 (const char *cls, const char* T, const char *als, const void *obj)
{
  g95x_pointer valo;
  int an = ! g95x_option.xml_defn;
  

  if (obj == NULL)
    g95x_die ("");

  add_lst1 ();

  valo = get_als_id (1, als, (g95x_pointer) obj);

  g95x_r_stack ();

  g95x_c_0.type = G95X_STACK_OBJ;
  g95x_c_0.name = cls;
  g95x_c_0.T    = T;

  if (pstack > 0)
    g95x_c_1.nelems++;

  indent (0);
  
  if (T && g95x_option.xml_type)
    {
      fprintf (g95x_out, "<%s%s-%s", prefix, T, cls)
    + ( an == 0 ? fprintf (" id=\"0x%x\"", valo) : 0 )
#ifdef G95X_ADD_ADDR
    + fprintf (g95x_out, " addr=\"%s-0x%x\"", als, obj)
#endif
    ;
    }
  else
    {
      fprintf (g95x_out, "<%s%s", prefix, cls)
    + ( an == 0 ? fprintf (" id=\"0x%x\"", valo) : 0 )
#ifdef G95X_ADD_ADDR
    + fprintf (g95x_out, " addr=\"%s-0x%x\"", als, obj)
#endif
    ;
    }

  if(T && !g95x_option.xml_type)
    g95x_add_att1_str (G95X_S_TYPE, T);

  if (g95x_option.xml_watch)
    g95x_add_att1_int ("watch-id", g95x_get_watch_id ());

  return 0;
}

int
g95x_add_obj1 (const char *cls, const char *T, int ni, const void *obj)
{
  g95x_pointer valo;
  int an = ! g95x_option.xml_defn;

  if (obj == NULL)
    g95x_die ("");

  add_lst1 ();

  if (an == 0) 
  {
    valo = get_obj_id (1, (g95x_pointer) obj, ni);
    if (ni == 0)
      g95x_die ("");
  }

  g95x_r_stack ();

  g95x_c_0.type = G95X_STACK_OBJ;
  g95x_c_0.name = cls;
  g95x_c_0.T    = T;

  if (pstack > 0)
    g95x_c_1.nelems++;

  indent (0);

  if (T && g95x_option.xml_type)
    {
      if ((an == 0) && (ni == 0))
        fprintf (g95x_out, "<%s%s-%s id=\"0x%x\"", prefix, T, cls, valo)
#ifdef G95X_ADD_ADDR
      + fprintf (g95x_out, " addr=\"0x%x\"", obj)
#endif
      ;
      else
        fprintf (g95x_out, "<%s%s-%s", prefix, T, cls);
    }
  else
    {
      if ((an == 0) && (ni == 0))
        fprintf (g95x_out, "<%s%s id=\"0x%x\"", prefix, cls, valo)
#ifdef G95X_ADD_ADDR
      + fprintf (g95x_out, " addr=\"0x%x\"", obj)
#endif
      ;
      else
        fprintf (g95x_out, "<%s%s", prefix, cls);
    }

  if (T && !g95x_option.xml_type)
    g95x_add_att1_str (G95X_S_TYPE, T);

  if (g95x_option.xml_watch)
    g95x_add_att1_int ("watch-id", g95x_get_watch_id ());

  return 0;
}


int
g95x_end_obj1 ()
{
  if (g95x_c_0.type != G95X_STACK_OBJ)
    g95x_die ("");

  if (g95x_c_0.nelems > 0)
    {
      indent (0);
      if (g95x_c_0.T && g95x_option.xml_type)
        fprintf (g95x_out, "</%s%s-%s>" G95X_CR, prefix, g95x_c_0.T, g95x_c_0.name);
      else
        fprintf (g95x_out, "</%s%s>" G95X_CR, prefix, g95x_c_0.name);
    }
  else
    {
      fprintf (g95x_out, "/>" G95X_CR);
    }

  g95x_l_stack ();

  return 0;
}

int
g95x_add_inc ()
{
  add_lst1 ();

  g95x_r_stack ();

  g95x_c_0.type = G95X_STACK_INC;
  g95x_c_0.name = "include";

  indent (0);

  return fprintf (g95x_out, G95X_CRBR "<%sinclude>", prefix, prefix, prefix);
}

int
g95x_end_inc ()
{
  if (g95x_c_0.type != G95X_STACK_INC)
    g95x_die ("");

  indent (0);
  fprintf (g95x_out, "</%s%s>", prefix, g95x_c_0.name);

  g95x_l_stack ();

  return 0;
}

int
g95x_add_fle (g95_file * f)
{
  g95x_pointer valo;


  valo = get_obj_id (1, (g95x_pointer) f, 0);

  g95x_r_stack ();

  g95x_c_0.type = G95X_STACK_FLE;
  g95x_c_0.name = "file";
  g95x_c_0.f = f;

  indent (0);

  fprintf (g95x_out, "<%sfile", prefix)
#ifdef G95X_ADD_ADDR
    + fprintf (g95x_out, " addr=\"0x%x\"", obj)
#endif
    + fprintf (g95x_out, " name=\"%s\" width=\"%d\" form=\"%s\">" G95X_BR,
	       f->filename, g95x_get_line_width (),
	       g95x_form_type (f->x.form), prefix);
  ;


  return 0;
}

int
g95x_end_fle ()
{
  if (g95x_c_0.type != G95X_STACK_FLE)
    g95x_die ("");

  indent (0);
  fprintf (g95x_out, "</%s%s>" G95X_CR, prefix, g95x_c_0.name);

  g95x_l_stack ();

  return 0;
}


int
g95x_add_att1_cst (const char *key, const char *vals)
{
  const char *s;

  fprintf (g95x_out, " %s=\"", key);

  for (s = vals; *s != 0; s++)
    {
      switch (*s)
	{
	case '"':
	  fprintf (g95x_out, "&quot;");
	  break;
	case '<':
	  fprintf (g95x_out, "&lt;");
	  break;
	case '>':
	  fprintf (g95x_out, "&gt;");
	  break;
	case '&':
	  fprintf (g95x_out, "&amp;");
	  break;
	case '\\':
	  fprintf (g95x_out, "\\\\");
	  break;
	default:
	  if (isprint (*s))
	    fprintf (g95x_out, "%c", *s);
	  else
	    fprintf (g95x_out, "&#%d;", (unsigned char) *s);
	}
    }

  fprintf (g95x_out, "\"");

  return 1;
}

int
g95x_add_att1_str (const char *key, const char *vals)
{
  return fprintf (g95x_out, " %s=\"%s\"", key, vals);
}

int
g95x_add_att1_int (const char *key, const int vali)
{
  if ((!strcmp (key, "c1") || (!strcmp (key, "c2"))))
    {
      if (vali == 0)
	g95x_die ("");
    }
  return fprintf (g95x_out, " %s=\"%d\"", key, vali);
}

int
g95x_add_att1_obj (const char *key, const void *valo)
{
  return fprintf (g95x_out, " %s=\"0x%x\"", key,
		  get_obj_id (0, (g95x_pointer) valo, 0))
#ifdef G95X_ADD_ADDR
    + fprintf (g95x_out, " addr-%s=\"0x%x\"", key, valo)
#endif
    ;
}

int
g95x_add_att1_als (const char *key, const char *als, const void *valo)
{
  return fprintf (g95x_out, " %s=\"0x%x\"", key,
		  get_als_id (0, als, (g95x_pointer) valo))
#ifdef G95X_ADD_ADDR
    + fprintf (g95x_out, " %s-addr=\"%s-0x%x\"", key, als, valo)
#endif
    ;
}

int
g95x_add_att1_und (const char *key)
{
  return fprintf (g95x_out, " %s=\"undef\"", key);
}

int
g95x_add_att1_chr (const char *key, char valc)
{
  return fprintf (g95x_out, " %s=\"%c\"", key, valc);
}

static int
end_att1 (int k)
{
  if (pstack + k < 0)
    g95x_die ("");
  if (g95x_c_stack (k).type != G95X_STACK_OBJ)
    g95x_die ("");
  if (g95x_c_stack (k).end++)
    return 0;
  return fprintf (g95x_out, ">" G95X_CR);
}

int
g95x_end_att1 ()
{
  return 0;
}



int
g95x_add_lst1 (const char *lst)
{

  g95x_r_stack ();

  g95x_c_0.type = G95X_STACK_LST;
  g95x_c_0.name = lst;

  return 1;
}


int
g95x_end_lst1 ()
{
  int p = 0;

  while (g95x_c_0.type == G95X_STACK_FLE)
    {
      g95x_close_file (g95x_c_0.f);
      p++;
    }

  if ((g95x_c_0.nelems > 0) || (p > 0))
    indent (0);

  if (g95x_c_0.type != G95X_STACK_LST)
    g95x_die ("");

  if ((g95x_c_0.nelems > 0) || (p > 0))
    fprintf (g95x_out, "</%s%s>" G95X_CR, prefix, g95x_c_0.name);

  g95x_l_stack ();

  return 0;
}


int
g95x_psh_lst1_obj (const void *valo)
{
  add_lst1 ();
  indent (1);

  g95x_c_0.nelems++;

  return fprintf (g95x_out, "<%sref val=\"0x%x\"", prefix,
		  get_obj_id (0, (g95x_pointer) valo, 0))
#ifdef G95X_ADD_ADDR
    + fprintf (g95x_out, " addr=\"0x%x\"", valo)
#endif
    + fprintf (g95x_out, "/>" G95X_CR);
}

int
g95x_psh_lst1_als (const char *als, const void *valo)
{
  add_lst1 ();
  indent (1);

  g95x_c_0.nelems++;

  return fprintf (g95x_out, "<%sref val=\"0x%x\"/>" G95X_CR, prefix,
		  get_als_id (0, als, (g95x_pointer) valo));
}

int
g95x_psh_lst1_chr (char valc)
{
  add_lst1 ();
  indent (1);

  g95x_c_0.nelems++;

  return fprintf (g95x_out, "<%sc val=\"%c\"/>" G95X_CR, prefix, valc);
}

int
g95x_psh_lst1_und ()
{
  add_lst1 ();
  indent (1);

  g95x_c_0.nelems++;

  return fprintf (g95x_out, "<%sundef/>" G95X_CR, prefix);
}

int
g95x_psh_lst1_str (const char *vals)
{
  add_lst1 ();
  indent (1);

  g95x_c_0.nelems++;

  return fprintf (g95x_out, "<%sstr val=\"%s\"/>" G95X_CR, prefix, vals);
}

int
g95x_psh_lst1_int (int vali)
{
  add_lst1 ();
  indent (1);

  g95x_c_0.nelems++;

  return fprintf (g95x_out, "<%sint val=\"%d\"/>" G95X_CR, prefix, vali);
}

int
g95x_add_sct1 (const char *sct)
{
  if (sct == NULL)
    g95x_die ("");

  if (g95x_c_0.type != G95X_STACK_INC)
    end_att1 (0);

  g95x_r_stack ();

  g95x_c_0.type = G95X_STACK_SCT;
  g95x_c_0.name = sct;

  if (pstack > 0)
    g95x_c_1.nelems++;

  indent (0);
  return fprintf (g95x_out, "<%s_%s_>" G95X_CR, prefix, sct);
}

int
g95x_end_sct1 ()
{
  indent (0);
  if (g95x_c_0.type != G95X_STACK_SCT)
    g95x_die ("");

  fprintf (g95x_out, "</%s_%s_>" G95X_CR, prefix, g95x_c_0.name);

  g95x_l_stack ();

  return 0;
}


int
g95x_add_txt1 ()
{
  int txt_indent = 0;

  switch (g95x_c_0.type)
    {
    case G95X_STACK_OBJ:
      end_att1 (0);
      txt_indent = 1;
      break;
    case G95X_STACK_LST:
      if (g95x_c_0.nelems > 0)
	{
	  txt_indent = 1;
	}
      else
	{
	  end_att1 (-1);
	  txt_indent = 0;
	}
      break;
    case G95X_STACK_FLE:
    case G95X_STACK_INC:
      txt_indent = 1;
      break;
    default:
      txt_indent = 0;
    }
  indent (txt_indent);
  return 0;
}

int g95x_put_txt1_cd = 1;

int
g95x_put_txt1 (const char *key, const char *txt, int cpp, int id, int *open)
{
  if (*open == 0)
    {
      *open = 1;
      g95x_add_txt1 ();
    }

  if (!strcmp (G95X_S_CODE, key)) 
  {
    if (!g95x_option.xml_sc)
      if ((g95x_put_txt1_cd == 0) && (cpp == 0) && txt && txt[0])
      {
        return fprintf (g95x_out, "%s", txt);
      }
    key = G95X_S_c;
  } 
  else if (!strcmp (G95X_S_COMMENT, key))
  {
    key = G95X_S_C;
  }
  else if (!strcmp (G95X_S_BR, key))
  {
    return fprintf (g95x_out, G95X_CRBR, prefix, prefix);
  }
  else if (!strcmp (G95X_S_STRING, key))
  {
    key = G95X_S_STR;
  }

  if (id == 0)
    {
      if ((!strcmp ("space", key)) && (!cpp) && txt && txt[0])
	return fprintf (g95x_out, "%s", txt);
      return
	(txt && txt[0]) ?
	cpp ?
	fprintf (g95x_out, "<%s%s-cpp>%s</%s%s-cpp>", prefix, key, txt, prefix,
		 key) : fprintf (g95x_out, "<%s%s>%s</%s%s>", prefix, key, txt,
				 prefix, key) : fprintf (g95x_out, "<%s%s/>",
							 prefix, key);
    }
  else
    {
      return
	(txt && txt[0]) ?
	cpp ?
	fprintf (g95x_out, "<%s%s-cpp cpp=\"0x%x\">%s</%s%s-cpp>", prefix, key, id,
		 txt, prefix, key) : fprintf (g95x_out,
					      "<%s%s id=\"0x%x\">%s</%s%s>",
					      prefix, key, id, txt, prefix,
					      key) : fprintf (g95x_out,
							      "<%s%s id=\"0x%x\"/>",
							      prefix, key,
							      id);
    }
  return 0;
}

int
g95x_end_txt1 (int *open)
{
  if (*open == 0)
    return 0;
  fprintf (g95x_out, G95X_CR);
  if (g95x_c_0.type == G95X_STACK_LST)
    if (g95x_c_0.nelems == 0)
      add_lst1 ();
  g95x_c_0.nelems++;
  *open = 0;
  return 0;
}
