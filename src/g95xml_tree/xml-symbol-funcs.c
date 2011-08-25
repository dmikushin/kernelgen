

static g95x_implicit_spec *head_is = NULL, *tail_is = NULL;

static void
g95x_push_is (g95_typespec * ts, g95x_locus * xw)
{
  g95x_implicit_spec *is;
  is = g95_get_implicit_spec ();
  is->ts = *ts;
  is->where = *xw;
  if (head_is == NULL)
    {
      tail_is = head_is = is;
    }
  else
    {
      tail_is->next = is;
      tail_is = is;
    }
}

static void
g95x_clear_is ()
{
  g95x_implicit_spec *is;
  for (is = head_is; is; is = head_is)
    {
      head_is = is->next;
      g95_free (is);
    }
  head_is = tail_is = NULL;
}

g95x_implicit_spec *
g95x_take_implicit_spec_list ()
{
  g95x_implicit_spec *is = head_is;
  head_is = tail_is = NULL;
  return is;
}
