
void g95x_resolve_epsilon( g95_expr *f, g95_expr *x ) {

    f->ts = x->ts;

}

void g95x_resolve_huge( g95_expr *f, g95_expr *x ) {

    f->ts = x->ts;

}

void g95x_resolve_tiny( g95_expr *f, g95_expr *x ) {

    f->ts = x->ts;

}
