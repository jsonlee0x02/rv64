// Math atan computing, scalar vsn with Newton-Raphson methods.
// Details see here:
// https://code.google.com/archive/p/math-neon/source/default/source>
#define M_PI_2 1.57079632679489661923 /* pi/2 */

// look up table
const float __atanf_lut[4] = {
    -0.0443265554792128,  // p7
    -0.3258083974640975,  // p3
    +0.1555786518463281,  // p5
    +0.9997878412794807   // p1
};

const float __atanf_pi_2 = M_PI_2;

float atanf_c(float x) {
    float a, b, r, xx;
    int m;

    union {
        float f;
        int i;
    } xinv, ax;

    ax.f = fabs(x);

    // fast inverse approximation (2x newton)
    xinv.f = ax.f;
    m = 0x3F800000 - (xinv.i & 0x7F800000);
    xinv.i = xinv.i + m;
    xinv.f = 1.41176471f - 0.47058824f * xinv.f;
    xinv.i = xinv.i + m;
    b = 2.0 - xinv.f * ax.f;
    xinv.f = xinv.f * b;
    b = 2.0 - xinv.f * ax.f;
    xinv.f = xinv.f * b;

    // if |x| > 1.0 -> ax = -1/ax, r = pi/2
    xinv.f = xinv.f + ax.f;
    a = (ax.f > 1.0f);
    ax.f = ax.f - a * xinv.f;
    r = a * __atanf_pi_2;

    // polynomial evaluation
    xx = ax.f * ax.f;
    a = (__atanf_lut[0] * ax.f) * xx + (__atanf_lut[2] * ax.f);
    b = (__atanf_lut[1] * ax.f) * xx + (__atanf_lut[3] * ax.f);
    xx = xx * xx;
    b = b + a * xx;
    r = r + b;

    // if x < 0 -> r = -r
    a = 2 * r;
    b = (x < 0.0f);
    r = r - a * b;

    return r;
}
