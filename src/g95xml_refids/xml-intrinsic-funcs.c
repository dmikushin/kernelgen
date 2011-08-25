
void g95x_get_intrinsics_info( 
    g95_intrinsic_sym** f,
    g95_intrinsic_sym** s,
    int* nf, int* ns
) {
    *f = functions;
    *s = subroutines;
    *nf = nfunc;
    *ns = nsub;
}


