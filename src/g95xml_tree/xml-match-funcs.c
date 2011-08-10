


/* common */

static g95x_voidp_list *common_list = NULL;

static void
g95x_common_push_list_symbol (g95_symbol * s)
{
  return g95x_push_list_symbol (&common_list, s);
}

static void
g95x_common_set_dimension (g95_symbol * s)
{
  return g95x_set_dimension (&common_list, s);
}

static void
g95x_common_push_list_common (g95_common_head * c)
{
  return g95x_push_list_common (&common_list, c);
}

static void
g95x_common_free_list ()
{
  return g95x_free_list (&common_list);
}

g95x_voidp_list *
g95x_common_take_list ()
{
  return g95x_take_list (&common_list);
}

static void
g95x_common_set_locus (g95x_locus * xw)
{
  if (common_list)
    {
      common_list->where = *xw;
    }
  else
    {
      g95x_die ("");
    }
}

/* enumerator */


static g95x_voidp_list *enumerator_list = NULL;

static void
g95x_enumerator_push_list_symbol (g95_symbol * s)
{
  return g95x_push_list_symbol (&enumerator_list, s);
}

static void
g95x_enumerator_free_list ()
{
  return g95x_free_list (&enumerator_list);
}

g95x_voidp_list *
g95x_enumerator_take_list ()
{
  return g95x_take_list (&enumerator_list);
}

static void
g95x_enumerator_set_locus (g95x_locus * xw)
{
  if (enumerator_list)
    {
      enumerator_list->where = *xw;
    }
  else
    {
      g95x_die ("");
    }
}

int
g95x_enum_bindc ()
{
  return enum_bindc;
}

/* namelist */

static g95x_voidp_list *naml_list = NULL;

static void
g95x_naml_push_list_symbol (g95_symbol * s)
{
  return g95x_push_list_symbol (&naml_list, s);
}

static void
g95x_naml_free_list ()
{
  return g95x_free_list (&naml_list);
}

g95x_voidp_list *
g95x_naml_take_list ()
{
  return g95x_take_list (&naml_list);
}

/* import */

static g95x_voidp_list *import_list = NULL;

static void
g95x_import_push_list_symbol (g95_symbol * s)
{
  return g95x_push_list_symbol (&import_list, s);
}

static void
g95x_import_free_list ()
{
  return g95x_free_list (&import_list);
}

g95x_voidp_list *
g95x_import_take_list ()
{
  return g95x_take_list (&import_list);
}

static void
g95x_import_set_locus (g95x_locus * xw)
{
  if (import_list)
    {
      import_list->where = *xw;
    }
  else
    {
      g95x_die ("");
    }
}
