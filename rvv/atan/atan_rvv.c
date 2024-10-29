vfloat32m2_t atanf_rvv(vfloat32m2_t x, size_t vl) {
    vfloat32m2_t a, b, r, xx, xinvf, axf;
    vint32m2_t ax_int, xinv_int;

    // Absolute value
    // ax.f = fabs(x);
    axf = vfabs_v_f32m2(x, vl);

    // Fast inverse approximation (2x Newton-Raphson)
    // xinv.f = ax.f;
    xinvf = axf;
    xinv_int = vreinterpret_v_f32m2_i32m2(xinvf);
    // m = 0x3F800000 - (xinv.i & 0x7F800000);
    vint32m2_t m = vsub_vv_i32m2(vmv_v_x_i32m2(0x3F800000, vl),
                                 vand_vv_i32m2(xinv_int, vmv_v_x_i32m2(0x7F800000, vl), vl), vl);
    // xinv.i = xinv.i + m;
    xinv_int = vadd_vv_i32m2(xinv_int, m, vl);
    xinvf = vreinterpret_v_i32m2_f32m2(xinv_int);
    // xinv.f = 1.41176471f - 0.47058824f * xinv.f;
    xinvf =
        vfsub_vv_f32m2(vfmv_v_f_f32m2(1.41176471f, vl), vfmul_vf_f32m2(xinvf, 0.47058824f, vl), vl);
    xinv_int = vreinterpret_v_f32m2_i32m2(xinvf);
    //  xinv.i = xinv.i + m;
    xinv_int = vadd_vv_i32m2(xinv_int, m, vl);
    xinvf = vreinterpret_v_i32m2_f32m2(xinv_int);
    // b = 2.0 - xinv.f * ax.f;
    b = vfsub_vv_f32m2(vfmv_v_f_f32m2(2.0f, vl), vfmul_vv_f32m2(xinvf, axf, vl), vl);
    // xinv.f = xinv.f * b;
    xinvf = vfmul_vv_f32m2(xinvf, b, vl);
    // b = 2.0 - xinv.f * ax.f;
    b = vfsub_vv_f32m2(vfmv_v_f_f32m2(2.0f, vl), vfmul_vv_f32m2(xinvf, axf, vl), vl);
    //  xinv.f = xinv.f * b;
    xinvf = vfmul_vv_f32m2(xinvf, b, vl);

    // If |x| > 1.0 -> ax = -1/ax, r = pi/2
    // xinv.f = xinv.f + ax.f;
    xinvf = vfadd_vv_f32m2(xinvf, axf, vl);
    // a = (ax.f > 1.0f);
    vbool16_t mask = vmfgt_vf_f32m2_b16(axf, 1.0f, vl);  // ax > 1.0
    vfloat32m2_t ones = vfmv_v_f_f32m2(1.0f, vl);
    vfloat32m2_t zeros = vfmv_v_f_f32m2(0.0f, vl);
    a = vmerge_vvm_f32m2(mask, zeros, ones, vl);
    // ax.f = ax.f - a * xinv.f
    axf = vfsub_vv_f32m2(axf, vfmul_vv_f32m2(a, xinvf, vl), vl);
    // r = a * __atanf_pi_2;
    r = vfmul_vf_f32m2(a, __atanf_pi_2, vl);
    // Polynomial evaluation
    // xx = ax.f * ax.f;
    xx = vfmul_vv_f32m2(axf, axf, vl);
    // a = (__atanf_lut[0] * ax.f) * xx + (__atanf_lut[2] * ax.f);
    a = vfadd_vv_f32m2(vfmul_vv_f32m2(vfmul_vf_f32m2(axf, __atanf_lut[0], vl), xx, vl),
                       vfmul_vf_f32m2(axf, __atanf_lut[2], vl), vl);
    // b = (__atanf_lut[1] * ax.f) * xx + (__atanf_lut[3] * ax.f);
    b = vfadd_vv_f32m2(vfmul_vv_f32m2(vfmul_vf_f32m2(axf, __atanf_lut[1], vl), xx, vl),
                       vfmul_vf_f32m2(axf, __atanf_lut[3], vl), vl);
    // xx = xx * xx;
    xx = vfmul_vv_f32m2(xx, xx, vl);
    // b = b + a * xx;
    b = vfadd_vv_f32m2(b, vfmul_vv_f32m2(a, xx, vl), vl);
    // r = r + b;
    r = vfadd_vv_f32m2(r, b, vl);

    // If x < 0 -> r = -r
    // a = 2 * r;
    a = vfmul_vf_f32m2(r, 2.0f, vl);
    mask = vmflt_vf_f32m2_b16(x, 0.0f, vl);  // x < 0
    // b = (x < 0.0f);
    b = vmerge_vvm_f32m2(mask, zeros, ones, vl);
    //  r = r - a * b;
    r = vfsub_vv_f32m2(r, vfmul_vv_f32m2(a, b, vl), vl);
    return r;
}
