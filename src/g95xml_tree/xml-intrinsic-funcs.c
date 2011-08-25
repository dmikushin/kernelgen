
void
g95x_get_intrinsics_info (g95_intrinsic_sym ** f,
			  g95_intrinsic_sym ** s, int *nf, int *ns)
{
  *f = functions;
  *s = subroutines;
  *nf = nfunc;
  *ns = nsub;
}

void
g95x_mark_intrinsic_used (g95_intrinsic_sym * is)
{
  int i;

  is->x.used = 1;
  if (is->id == G95_ISYM_EXTENSION)
    return;
  for (i = 0; i < nfunc; i++)
    if (functions[i].id == is->id)
      functions[i].x.used = 1;
  for (i = 0; i < nsub; i++)
    if (subroutines[i].id == is->id)
      subroutines[i].x.used = 1;
}
