#include "g95.h"

#include <stdio.h>
#include <stdlib.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

#include <string.h>

static char* Txt = NULL;
static int done = 0;
static int example_on = 0;
static char EXAMPLE[256];

static const char 
  * EX1 = "!EXAMPLE:",
  * EX2 = "!END";

int g95x_xml_example_switch (const char * x)
{
  if (strncmp (x, EX1, strlen (EX1)) == 0)
    {
      example_on = 1;
      strcpy (EXAMPLE, x + strlen (EX1));
    }
  if (strncmp (x, EX2, strlen (EX2)) == 0)
    {
      example_on = 0;
      EXAMPLE[0] = '\0';
    }
  return 0;
}



int g95x_xml_example (const char* txt)
{
  const char * objN, * objT;

  if (!example_on)
    return 0;

  if (Txt == NULL)
    {
      const char * f = g95x_get_f95_file ();
      FILE* fp = fopen (f, "r");
      struct stat st;
      if (fp == NULL)
        g95x_die ("Cannot open `%f' for reading", f);
      stat (f, &st);
      Txt = g95_getmem (st.st_size+1);
      fread (Txt, st.st_size, 1, fp);
      fclose (fp);
      Txt[st.st_size] = '\0';
    }

  if (strcmp (Txt, txt))
    return 0;

  if (!done)
    {

      objN = g95x_current_obj_name ();
      objT = g95x_current_obj_type ();

      g95x_add_att1_str ("example-id", EXAMPLE);
      example_on = 0;

    }

  done++;

  return 0;
}

static FILE* wout = NULL;

const char * g95x_watch (const char * x)
{
#define tt(x) ((x) + strlen (x) + 1)
  const char * t1 = x;
  const char * t2 = tt(t1);
  const char * t3 = tt(t2);
  const char * t4 = tt(t3);
  fprintf (wout, "  %s obj=\"%d\"%s%s%s\n", t1, g95x_get_watch_id (), t2, t3, t4);
  return t3;
}

int g95x_add_watch ()
{
  int i, n;
  char F[512];
  memset (&F[0], '\0', sizeof (F));
  strcpy (&F[0], g95x_get_xml_file ());
  strcat (&F[0], ".watch");
  wout = fopen (&F[0], "w");
  fprintf (wout, "<strings>\n");
  return 0;
}

int g95x_end_watch ()
{
  fprintf (wout, "</strings>\n");
  fclose (wout);
  wout = NULL;
  return 0;
}

static int watch_id = 0;

int g95x_get_watch_id ()
{
  return watch_id;
}


int g95x_new_watch_id ()
{
  return ++watch_id;
}


