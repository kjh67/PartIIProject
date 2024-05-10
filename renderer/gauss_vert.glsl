#version 450

// Per-gaussian inputs
in vec3 center;
in vec4 rotation;
in vec3 scale;

in vec3 shs;
in float opacity;

// Per-vertex inputs
in vec2 position;

//Uniform inputs
uniform mat4 projection;
uniform mat4 view;
uniform vec2 tanfovxy;
uniform vec2 focal;
uniform vec2 viewport;


// Outputs to the fragment shader, per vertex
out vec2 gauss_center;
out vec3 gauss_conic;

out vec3 gauss_colour;
out float gauss_opacity;


//Define SH coefficients for colour calculation - taken from reference implementation
#define SHC_0 0.28209479177387814f

#define SHC_1 0.4886025119029199f

#define SHC_2_0 1.0925484305920792f
#define SHC_2_1 -1.0925484305920792f
#define SHC_2_2 0.31539156525252005f
#define SHC_2_3 -1.0925484305920792f
#define SHC_2_4 0.5462742152960396f

#define SHC_3_0 -0.5900435899266435f
#define SHC_3_1 2.890611442640554f
#define SHC_3_2 -0.4570457994644658f
#define SHC_3_3 0.3731763325901154f
#define SHC_3_4 -0.4570457994644658f
#define SHC_3_5 1.445305721320277f
#define SHC_3_6 -0.5900435899266435f


mat3 compute_covariance_3D() {
    mat3 S = mat3(scale.x, 0, 0, 0, scale.y, 0, 0, 0, scale.z);
    
    vec4 norm_rot = rotation;// / normalize(rotation);

    float q0 = norm_rot.x;
    float q1 = norm_rot.y;
    float q2 = norm_rot.z;
    float q3 = norm_rot.w;
    // R calculated using formula from wikipedia
    mat3 R = mat3(
        1 - 2*(q2*q2 + q3*q3), 2*(q1*q2 - q0*q3), 2*(q1*q3 + q0*q2),
        2*(q1*q2 + q0*q3), 1 - 2*(q1*q1 + q3*q3), 2*(q2*q3 - q0*q1),
        2*(q1*q3 - q0*q2), 2*(q2*q3 + q0*q1), 1 - 2*(q1*q1 + q2*q2)
    );

    // sigma calculated using equation given in paper
    mat3 sigma = transpose(R) * transpose(S) * S * R;

    return sigma;
}

vec3 compute_covariance_2D() {
    // t is midpoint of the gaussian transformed by view matrix
    vec4 t = view * vec4(center, 1);

    //Use equations from that paper
    float tx_tz = t.x / t.z;
    float ty_tz = t.y / t.z;

    // should be 1.3 times tanfovx/y
    vec2 limits = 1.3 * tanfovxy;
    t.x = min(limits.x, max(-limits.x, tx_tz)) * t.z;
    t.y = min(limits.y, max(-limits.y, ty_tz)) * t.z;

    mat3 J = mat3(
        focal.x / t.z, 0.0, -(focal.x * t.x) / (t.z*t.z),
        0.0, focal.y / t.z, -(focal.y * t.y) / (t.z*t.z),
        0.0,0.0,0.0
    );
    //mat3 W = transpose(mat3(view));
    mat3 W = mat3(view[0][0], view[1][0], view[2][0], 
                  view[0][1], view[1][1], view[2][1],
                  view[0][2], view[1][2], view[2][2]);

    mat3 T = W * J;

    mat3 cov3D = compute_covariance_3D();
    // Equation 31 from the papers
    mat3 cov2D = transpose(T) * transpose(cov3D) * T;

    return vec3(cov2D[0][0] + 0.3, cov2D[0][1], cov2D[1][1] + 0.3);
}

//TODO: implement full sh from colour function
vec3 colour_from_harmonics() {
    // camera assumed to be at origin; viewing direction will be normalised gauss center position
    vec3 view_dir = normalize(center);

    float x = view_dir.x;
    float y = view_dir.y;
    float z = view_dir.z;

    // 0th order harmonics
    vec3 result = SHC_0 * shs;
    //vec3 result = vec3(0.5,0.5,0.5);
    //vec3 random = shs;

    // 1st order harmonics
    // result = result - SHC_1 * y * shs[1]
    //                 + SHC_1 * z * shs[2]
    //                 - SHC_1 * x * shs[3];

    // // 2nd order harmonics
    // float xx = x*x; float yy = y*y; float zz = z*z;
    // float xy = x*y; float yz = y*z; float xz = x*z;

    // result = result + SHC_2_0 * xy * (shs[12], shs[13], shs[14])
    //                 + SHC_2_1 * yz * (shs[15], shs[16], shs[17])
    //                 + SHC_2_2 * (2*zz - xx - yy) * (shs[18], shs[19], shs[20])
    //                 + SHC_2_3 * xz * (shs[21], shs[22], shs[23])
    //                 + SHC_2_4 * (xx - yy) * (shs[24], shs[25], shs[26]);


    // float r = SHC_0 * shs.r + 0.5;
    // float g = SHC_0 * shs.g + 0.5;
    // float b = SHC_0 * shs.b + 0.5;

    result += 0.5f;

    return result;
}



void main() {
    // OpenGL uses a right-handed rather than left-handed coordinate system; flip the z axis direction
    vec4 camspace = view * vec4(center, 1);
    vec4 pos2d = projection * camspace;
    vec3 cov2D = compute_covariance_2D();

    float det_cov2D = cov2D.x*cov2D.z - cov2D.y*cov2D.y;
    gauss_conic = vec3(cov2D.z, -cov2D.y, cov2D.x) / det_cov2D;

    float mid = 0.5 * (cov2D.x + cov2D.z);
    float l1 = mid + sqrt(max(0.1,mid*mid - det_cov2D));
    float l2 = mid - sqrt(max(0.1,mid*mid - det_cov2D));
    vec2 v1 = 7.0 * sqrt(l1) * normalize(vec2(cov2D.y, l1 - cov2D.x));
    vec2 v2 = 7.0 * sqrt(l2) * normalize(vec2(cov2D.x - l1, cov2D.y));

    gauss_center = vec2(pos2d) / pos2d.w;
    vec2 gauss_position = vec2(gauss_center + position.x * (position.y < 0.0 ? v1 : v2) / viewport);

    gauss_colour = colour_from_harmonics();
    gauss_opacity = opacity;

    gl_Position = vec4(gauss_position, pos2d.z / pos2d.w, 1);
}
