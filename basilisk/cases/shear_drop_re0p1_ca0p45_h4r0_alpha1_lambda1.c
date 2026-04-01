/* 2D shear droplet benchmark for Basilisk.
   Re = 0.1, Ca = 0.45, H = 4 R0, alpha = 1, lambda = 1.
   We choose R0 = 1 and gamma_dot = 1, so the walls are at y = +/- 2
   with tangential velocities u = +/- 2. */

#include "navier-stokes/centered.h"
#include "two-phase.h"
#include "tension.h"

#define RE 0.1
#define CA 0.45
#define WE (RE*CA)
#define R0 1.
#define H 4.
#define T_END 7.

int MAXLEVEL = 7;
int MINLEVEL = 7;

u.t[top] = dirichlet (y);
u.t[bottom] = dirichlet (y);

int main() {
  L0 = H;
  origin (-L0/2., -L0/2.);
  periodic (right);
  DT = 0.02;

  rho1 = 1.;
  rho2 = 1.;
  mu1 = 1./RE;
  mu2 = 1./RE;
  f.sigma = 1./WE;

  init_grid (1 << MINLEVEL);
  run();
}

event init (i = 0) {
  fraction (f, sq(R0) - sq(x) - sq(y));
  foreach() {
    u.x[] = y;
    u.y[] = 0.;
  }
  FILE * fp = fopen ("interface_t00.00.dat", "w");
  output_facets (f, fp);
  fclose (fp);
}

event adapt (i++) {
  adapt_wavelet ((scalar *){f, u}, (double[]){1e-3, 5e-4, 5e-4}, MAXLEVEL, MAXLEVEL);
}

event deformation (t += 0.05) {
  double sum_f = 0., sum_fx = 0., sum_fy = 0.;
  foreach (reduction(+:sum_f) reduction(+:sum_fx) reduction(+:sum_fy)) {
    if (f[] > 1e-12) {
      double dvf = dv()*f[];
      sum_f += dvf;
      sum_fx += x*dvf;
      sum_fy += y*dvf;
    }
  }

  double xc = sum_fx/sum_f;
  double yc = sum_fy/sum_f;
  double rmax = -HUGE, rmin = HUGE;

  foreach (reduction(max:rmax) reduction(min:rmin)) {
    if (f[] > 0. && f[] < 1.) {
      coord p;
      coord n = mycs (point, f);
      double alpha = plane_alpha (f[], n);
      plane_area_center (n, alpha, &p);
      double xi = x + Delta*p.x - xc;
      double yi = y + Delta*p.y - yc;
      double r = sqrt (sq(xi) + sq(yi));
      if (r > rmax)
        rmax = r;
      if (r < rmin)
        rmin = r;
    }
  }

  double D = (rmax - rmin)/(rmax + rmin);
  fprintf (stderr, "%.8f %.12e %.12e %.12e %.12e %.12e\n",
           t, xc, yc, rmin, rmax, D);
}

event interfaces (t += 1.) {
  char name[128];
  sprintf (name, "interface_t%05.2f.dat", t);
  FILE * fp = fopen (name, "w");
  output_facets (f, fp);
  fclose (fp);
}

event final_interface (t = T_END) {
  char name[128];
  sprintf (name, "interface_t%05.2f.dat", t);
  FILE * fp = fopen (name, "w");
  output_facets (f, fp);
  fclose (fp);
}

event stop (t = T_END) {
}
