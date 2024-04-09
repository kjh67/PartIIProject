#version 450

/*Inputs from vertex shader*/
in vec2 gauss_center;
in vec3 gauss_conic;

in vec3 gauss_colour;
in float gauss_opacity;

/*Constants*/
uniform vec2 viewport;
uniform vec2 focal;

void main() {
    //gl_FragColor = vec4(gauss_colour, gauss_opacity);
    vec2 d = (gauss_center - 2.0 * (gl_FragCoord.xy/viewport - vec2(0.5, 0.5))) * viewport * 0.5;
	float power = -0.5 * (gauss_conic.x * d.x * d.x + gauss_conic.z * d.y * d.y) - gauss_conic.y * d.x * d.y;
	if (power > 0.0) discard;
	float alpha = min(0.99, gauss_opacity * exp(power));
	if(alpha < 5./255.) discard;

	gl_FragColor = vec4(gauss_colour, alpha); // multiply gauss_colour by alpha if need be

}