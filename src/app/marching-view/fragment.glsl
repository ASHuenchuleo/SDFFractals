#define MAX_DIR_LIGHTS 0
#define MAX_POINT_LIGHTS 2
#define MAX_SPOT_LIGHTS 0
#define MAX_HEMI_LIGHTS 0
#define MAX_SHADOWS 0

uniform vec2 resolution;
uniform vec3 colorA;
uniform vec3 colorB;
uniform vec3 camera;
uniform vec3 target;
uniform float fov;
uniform float frame;

uniform int scene;
uniform int maxIters;
uniform bool displace;
uniform float tolerance;
uniform float stepMultiplier;
uniform bool tiled;
uniform float drawDistance;
uniform float worldScale;
uniform bool shadows;

uniform float pointLightDistance[MAX_POINT_LIGHTS];

/* Enum */
const int SIMPLE = 1;
const int INFINITE = 2;
const int INTERSECTION = 3;
const int LIGHTBULB = 4;
const int PYRAMID = 5;
const int SPONGE = 6;
const int MANDELBULB = 7;


/* Basic operations */
float dot2( in vec2 v ) { return dot(v,v); }
float dot2( in vec3 v ) { return dot(v,v); }
float ndot( in vec2 a, in vec2 b ) { return a.x*b.x - a.y*b.y; }

/* globals */
int fracIterations = 9;
float niters = 0.;



/////////////////////////////////////////////////////////////////////////

mat3 rotationMatrix3(vec3 axis, float angle)
{
  axis = normalize(axis);
  float s = sin(angle);
  float c = cos(angle);
  float oc = 1.0 - c;

  return mat3(oc * axis.x * axis.x + c,           oc * axis.x * axis.y - axis.z * s,  oc * axis.z * axis.x + axis.y * s,
  oc * axis.x * axis.y + axis.z * s,  oc * axis.y * axis.y + c,           oc * axis.y * axis.z - axis.x * s,
  oc * axis.z * axis.x - axis.y * s,  oc * axis.y * axis.z + axis.x * s,  oc * axis.z * axis.z + c          );
}

/////////////////////////////////////////////////////////////////////////

/* Distance functions */
// Some from https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float sdSphere(vec3 p, float r) {
  return sqrt( pow(p.x, 2.f) + pow(p.y, 2.f) + pow(p.z, 2.f)) - r ;
}

float sdBox( vec3 p, vec3 b )
{
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}

float sdCappedCylinder( vec3 p, float h, float r )
{
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(h,r);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}

float sdPlane( vec3 p , vec3 n) { return dot(p, n); }
float sdPlane( vec3 p , vec4 n) { return dot(p, n.xyz) - n.w; }


float hash(in float n)
{
  return fract(sin(n)*43758.5453123);
}

float noise(in vec2 x)
{
  vec2 p = floor(x);
  vec2 f = fract(x);
  f = f*f*(3.0-2.0*f);
  float n = p.x + p.y*57.0;
  float res = mix(mix( hash(n+  0.0), hash(n+  1.0),f.x),
  mix( hash(n+ 57.0), hash(n+ 58.0),f.x),f.y);
  return res;
}

//operations
// https://iquilezles.org/www/articles/distfunctions/distfunctions.htm
float opUnion( float d1, float d2 ) { return min(d1,d2); }

float opSubtraction( float d1, float d2 ) { return max(-d1,d2); }

float opIntersection( float d1, float d2 ) { return max(d1,d2); }

float opSmoothUnion( float d1, float d2, float k ) {
  float h = clamp( 0.5 + 0.5*(d2-d1)/k, 0.0, 1.0 );
  return mix( d2, d1, h ) - k*h*(1.0-h); }

float opSmoothSubtraction( float d1, float d2, float k ) {
  float h = clamp( 0.5 - 0.5*(d2+d1)/k, 0.0, 1.0 );
  return mix( d2, -d1, h ) + k*h*(1.0-h); }

float opSmoothIntersection( float d1, float d2, float k ) {
  float h = clamp(0.5 - 0.5*(d2-d1)/k, 0.0, 1.0);
  return mix(d2, d1, h) + k*h*(1.0-h);
}

// displaces the input vecotr
vec3 opdisplace(vec3 p, float k){
  float c = cos(k + p.y);
  float s = sin(k + p.y);
  mat2  m = mat2(c,-s,s,c);
  vec3  q = vec3(m*p.xz,p.y);
  return q;
}
// Finite repitition
vec3 opRepLim( in vec3 p, in float c, in vec3 l)
{
  vec3 q = p-c*clamp(round(p/c),-l,l);
  return q ;
}

// Infinite repetition
vec3 opRep( vec3 p, vec3 c)
{
  vec3 q =  mod( p, c ) - .5 * c;
  return q;
}

float opDisplace( vec3 p)
{
  float dis = sin(2.*sin(frame/60.) * p.x)*sin(sin(2.*frame/60. + 52.) * p.y)*sin(2.*sin(frame/60. + 162.) * p.z);
  return dis;
}

//http://www.pouet.net/topic.php?post=367360
const vec3 pa = vec3(1., 57., 21.);
const vec4 pb = vec4(0., 57., 21., 78.);
float perlin(vec3 pos, vec4 quat ) {
  mat3 transform = rotationMatrix3( quat.xyz, quat.w );
  vec3 p = pos * transform;
  vec3 i = floor(p);
  vec4 a = dot( i, pa ) + pb;
  vec3 f = cos((p-i)*acos(-1.))*(-.5)+.5;
  a = mix(sin(cos(a)*a),sin(cos(1.+a)*(1.+a)), f.x);
  a.xy = mix(a.xz, a.yw, f.y);
  return mix(a.x, a.y, f.z);
}
float perlin(vec3 p ) {
  vec3 i = floor(p);
  vec4 a = dot( i, pa ) + pb;
  vec3 f = cos((p-i)*acos(-1.))*(-.5)+.5;
  a = mix(sin(cos(a)*a),sin(cos(1.+a)*(1.+a)), f.x);
  a.xy = mix(a.xz, a.yw, f.y);
  return mix(a.x, a.y, f.z);
}




// https://github.com/stackgl/glsl-look-at/blob/gh-pages/index.glsl

mat3 calcLookAtMatrix(vec3 origin, vec3 target, float roll) {
  vec3 rr = vec3(sin(roll), cos(roll), 0.0);
  vec3 ww = normalize(target - origin);
  vec3 uu = normalize(cross(ww, rr));
  vec3 vv = normalize(cross(uu, ww));
  return mat3(uu, vv, ww);
}

// https://github.com/stackgl/glsl-camera-ray

vec3 rayDirection(vec2 coord){
  return normalize( vec3( coord, 1. ) );
}

vec3 getRay(mat3 camMat, vec2 screenPos, float lensLength) {
  return normalize(camMat * vec3(screenPos, lensLength));
}
vec3 getRay(vec3 origin, vec3 target, vec2 screenPos, float lensLength) {
  mat3 camMat = calcLookAtMatrix(origin, target, 0.0);
  return getRay(camMat, screenPos, lensLength);
}

vec2 getScreenPos(void ){
  //1 : retrieve the fragment's coordinates
  vec2 uv = ( gl_FragCoord.xy / resolution.xy ) * 2.0 - 1.0;
  //preserve aspect ratio
  uv.x *= resolution.x / resolution.y;
  return uv;
}

float sdfSimple(vec3 pos){ ;
  float sphereDis = sdSphere(pos, 1.f);
  vec3 boxDims = vec3(.5f, .5f, .5f);
  vec3 boxPos = vec3(2.f, 0.f, 0.f) * sin(frame/60.) ;
  mat3 rotation = rotationMatrix3(vec3(.5f, .5f, 0.f), frame/120.);
  float boxDis = sdBox(pos * rotation - boxPos, boxDims);

  vec3 repPos = opRep(pos, vec3(20.f, 20.f, 20.f));
  float boxRepDis = sdBox(repPos * rotation - boxPos, boxDims);


  return opSmoothUnion(sphereDis, boxDis, .5f);
}

float sdfIntersection(vec3 pos){


  vec3 boxDims = vec3(1.f, 1.f, 1.f);
  vec3 boxPos = vec3(1.f, 0.f, 0.f) * sin(frame/60.) ;
  mat3 rotation = rotationMatrix3(vec3(.5f, .5f, 0.f), frame/20.);

  float boxDis = sdBox(pos * rotation, boxDims);
  float sphereDis = sdSphere(pos - boxPos, 1.f);


  return opSmoothIntersection(boxDis, sphereDis, .5f);
}


float sdfInfinite(vec3 pos){

  vec3 r = vec3( 3., 0., 3. );
  vec3 q = opRep( pos, r );

  pos.xz += frame/20. *.5;
  float n = noise( ( .75 * pos.xz / r.xz ) );

  float s = max( .5, n * 1.125 );
  float h = n * 5.;

  float pl = sdPlane( pos, vec4( 0.,1.,0., h * .9 ) );
  float rb = sdBox( q, vec3( s, h, s )) - (.85 + ( .5+sin( frame/60. + n )*.5 ) * .1);

  return opSmoothUnion(pl, rb, 0.5f);
}

// https://github.com/nicoptere/raymarching-for-THREE
float sdfLightbulb(vec3 pos){

  //create a 5 units radius sphere
  float sph = sdSphere( pos, 5. );

  //create a 10 units high, 4 units radius cylinder and positions it at Y = -12.5 units
  float cyl = sdCappedCylinder( pos - vec3( 0.,-12.5,0.) , 10., 4.);

  //stores a copy of the position being evaluated
  vec3 nPos = pos * .45;

  //adds some delta
  nPos.y -= frame/60. * .05;

  //creates a transform (time-based rotation about the Y axis)
  vec4 quat = vec4( 0., 1., 0., -frame/60. * .1 );

  //evaluates a noise field using the transform above (the noise field "rotates")
  float noi = max( -.5, .5-abs( perlin( nPos, quat ) ) );

  //combines the shapes:
  // 1 - blend the sphere and the cylinder: smin( sph, cyl, .99 )
  // 2 - return the intersection of the blended shapes with the noise field
  return opIntersection( opSmoothUnion( sph, cyl, .99 ), noi );

}

float sdfPyramid(vec3 pos){
  vec3 a1 = vec3(1,1,1);
  vec3 a2 = vec3(-1,-1,1);
  vec3 a3 = vec3(1,-1,-1);
  vec3 a4 = vec3(-1,1,-1);
  vec3 c;
  int n = 0;
  float dist, d;
  float Scale = 2.f;
  float Offset = 1.f;
  float r;
  pos = pos * rotationMatrix3(vec3(.5f, .5f, 0.f), frame/120.);
  while (n < fracIterations) {
    pos = pos * rotationMatrix3(vec3(.5f, .5f, 0.f), radians(frame));

    if(pos.x+pos.y<0.) pos.xy = -pos.yx; // fold 1
    if(pos.x+pos.z<0.) pos.xz = -pos.zx; // fold 2
    if(pos.y+pos.z<0.) pos.zy = -pos.yz; // fold 3
    pos = pos*Scale - Offset*(Scale-1.0);

    n++;
  }
  return (length(pos) ) * pow(Scale, -float(n));
}

float sdfMenger3(vec3 pos){
  vec3 C = vec3(2.f, 4.8f, 0.f);
  C.x += 2.*sin(frame/60.);
  float r = pow(length(pos), 2.);


  float scale = 1.3f;
  float x1, y1, z1;
  int i;
  pos = pos * rotationMatrix3(vec3(.5f, .5f, 0.f), frame/120.);
  for(i = 0; i < fracIterations && r < 100.f; i++){
    pos = pos * rotationMatrix3(vec3(.0f, .1f, 0.f), radians(28.));

    pos.x=abs(pos.x);
    pos.y=abs(pos.y);
    pos.z=abs(pos.z);
    if(pos.x - pos.y < 0.){x1=pos.y;pos.y=pos.x;pos.x=x1;}
    if(pos.x - pos.z < 0.){x1=pos.z;pos.z=pos.x;pos.x=x1;}
    if(pos.y - pos.z < 0.){y1=pos.z;pos.z=pos.y;pos.y=y1;}

    pos = pos * rotationMatrix3(vec3(.0f, .1f, 0.f), sin(radians(frame/2.)));


    pos.xy = scale * pos.xy-C.xy*(scale-1.);
    pos.z=scale*pos.z;
    if(pos.z>0.5*C.z*(scale-1.)) pos.z-=C.z*(scale-1.);

    r= pow(length(pos), 2.);
  }
  niters = float(i);
  return (length(pos) - 2.) * pow(scale, -float(i));
}

// http://blog.hvidtfeldts.net/index.php/2011/09/distance-estimated-3d-fractals-v-the-mandelbulb-different-de-approximations/
// http://www.fractalforums.com/index.php?topic=16793.msg64299#msg64299
float sdfMandelbulb(vec3 z0){
  vec4 z = vec4(z0,1.0), c = z;
  float r = length(z.xyz),zr,theta,phi,p=8.;//p is the power
  phi = atan(z.y, z.x) * p;// th = atan(z.y, z.x) + phase.x; ...and here
  theta = asin(z.z / r) * p;// ph = acos(z.z / r) + phase.y; add phase shifts here

  vec2 phase = vec2(sin(radians(60. + frame/20.0 )), sin(radians(60. - frame/5.0 )));
  int n;
  for (int i = 0; n < fracIterations; n++) {
    zr = pow(r, p-1.0);
    z=zr*vec4(r*vec3(sin(theta)*vec2(cos(phi),sin(phi)),cos(theta)),z.w*p)+c; // this version was from the forums
    r = length(z.xyz);
    if (r > 100.f) break;
    phi = (atan(z.y, z.x) + phase.x) * p;// th = atan(z.y, z.x) + phase.x; ...and here
    theta = (acos(z.z / r) + phase.y) * p;// ph = acos(z.z / r) + phase.y; add phase shifts here
  }
  niters = float(n);
  return 0.5*log(r)*r/z.w;
}

float sdf(vec3 ip){
  if(tiled)
    ip = opRepLim(ip, 15.f, vec3(5.f, 5.f, 5.f));

  ip = ip/worldScale;

  float dis = 0.;
  if (scene == SIMPLE)
    dis = sdfSimple(ip);
  else if (scene == INFINITE)
    dis = sdfInfinite( ip );
  else if (scene == INTERSECTION)
    dis = sdfIntersection( ip );
  else if (scene == LIGHTBULB)
    dis = sdfLightbulb( ip );
  else if (scene == PYRAMID)
    dis = sdfPyramid( ip );
  else if (scene == SPONGE)
    dis = sdfMenger3( ip );
  else if (scene == MANDELBULB)
    dis = sdfMandelbulb( ip );

  if(displace)
    dis += opDisplace(ip);

  return dis;
}

// https://github.com/nicoptere/raymarching-for-THREE/blob/master/raymarcher.js
vec4 raymarch( vec3 pos, vec3 dir){


  //2 : camera position and ray direction


  //3 : ray march loop
  //ip will store where the ray hits the surface
  vec3 ip;

  float temp;
  //variable step size
  float t = 0.0;
  int i = 0;
  float limit = tolerance;
    for( i = 0; i < maxIters; i++) {

      //update position along path
      ip = pos + dir * t;

      if (t > drawDistance)
        break;

      temp = sdf(ip);

      // If im not close enough I do a smaller step. Prevents tearing.
      if( temp > limit ) temp *= stepMultiplier;
      //increment the step along the ray path
      t += temp;

      //break the loop if the distance was too small
      //this means that we are close enough to the surface

      if( temp < limit ) break;

      // The detail depends on the zoom level
      limit = min(max(tolerance,t/pow(10., 4.5)), 0.01f);

    }
  // http://2008.sub.blue/blog/2009/9/20/quaternion_julia.html

  vec4 ans = vec4(ip - pos, float(i));
  return ans;
}
/* Visualizacion */
//https://github.com/stackgl/glsl-sdf-normal
vec3 calcNormal(vec3 pos, float eps) {
  const vec3 v1 = vec3( 1.0,-1.0,-1.0);
  const vec3 v2 = vec3(-1.0,-1.0, 1.0);
  const vec3 v3 = vec3(-1.0, 1.0,-1.0);
  const vec3 v4 = vec3( 1.0, 1.0, 1.0);

  vec3 norm;
  norm = normalize( v1 * sdf( pos + v1*eps ) +
    v2 * sdf( pos + v2*eps ) +
    v3 * sdf( pos + v3*eps ) +
    v4 * sdf( pos + v4*eps ) );
  return norm;
}

vec3 calcNormal(vec3 pos) {
  return calcNormal(pos, 0.0001);
}

vec4 rimlight( vec3 pos, vec3 nor )
{
  vec3 v = normalize(-pos);
  float vdn = 1.0 - max(dot(v, nor), 0.0);
  return vec4( vec3( smoothstep(0., 1.0, vdn) ), 1.);
}

vec4 shading( vec3 pos, vec3 nor, vec3 rd, vec3 diffuse, float nsteps)
{
  float specularHardness = 128.;
  float specular = 1.;
  float ambientFactor = 0.0005;

  vec4 addedLights = vec4(0.0, 0.0, 0.0, 1.0);
  float depth;

  vec3 pointLightColor[MAX_POINT_LIGHTS];
  vec3 pointLightPosition[MAX_POINT_LIGHTS];

  pointLightPosition[0] = vec3(1.f, 20.f , -1.f);
  pointLightColor[0] = colorA;

  pointLightPosition[1] = vec3(-1.f, 15.f , 1.f);
  pointLightColor[1] = colorB;

  diffuse += abs(nor) * 0.05f; // Cooler colors
  addedLights +=  vec4(colorB * nsteps/150., 0.5) ; // Glow


  float aoStrength = 1.3;
  float ao = 1.0 - clamp((nsteps / float(150.)) * aoStrength, 0.0, 0.5);
  diffuse *= ao;
  addedLights.rgb += diffuse;

  for (int l = 0; l < MAX_POINT_LIGHTS; l++) {
    vec3 lightDirection = -normalize( pointLightPosition[l] ); // Direccion
    float dist = length(pointLightPosition[l] - pos);
    float depth = 1./dist;
    float projection =  clamp(dot( -lightDirection, nor), 0., 1.);

    vec3 lightColor =  projection * pointLightColor[l] * depth;
    addedLights.rgb +=  lightColor ;

    float spec = pow(max(0.0, projection), specularHardness) * specular;
    addedLights.rgb += spec * pointLightColor[l];
  }

    for (int l = 0; l < 2; l++) {
      vec3 lightDirection = -normalize(pointLightPosition[l]);

      float shadowsStrength = .1f ;
      if (shadows) {
        // The shadow ray will start at the intersection point and go
        // towards the point light.  We initially move the ray origin
        // a little bit along this direction so that we don't mistakenly
        // find an intersection with the same point again.

        vec3 rO = pos + nor * 0.1f * 2.0;
        vec4 result = raymarch(rO, -lightDirection);
        float dist = length(result.xyz);


        // Again, if our estimate of the distance to the set is small, we say
        // that there was a hit.  In this case it means that the point is in
        // shadow and should be given darker shading.
          // (darkening the shaded value is not really correct, but looks good)
        if (dist < 100.f)
          addedLights.rgb *= 1.0 - shadowsStrength;
      }


    }


  return addedLights;
}

// Luces de https://stackoverflow.com/questions/30151086/threejs-how-do-i-make-a-custom-shader-be-lit-by-the-scenes-lights
void main( void ) {
  vec2 uv = getScreenPos();

  vec3 pos = camera;
  vec3 dir = getRay(camera, target, uv, fov);

  vec4 marchResult = raymarch(pos, dir);

  vec3 ip = marchResult.xyz;
  float nsteps = marchResult.w;

  float dis = length(ip);

  // Background
  gl_FragColor = vec4( mix( colorA, colorB, sin( uv.y + 1.5 ) ), 1. );
  if (dis < drawDistance){
    // Global position of the collided object
    vec3 pos = camera + ip;
    //diffuse color
    //vec3 diffuse = vec3(.87f, .828f, .71f);
    float lambda = 1. - niters / (float(fracIterations) + 5.);
    vec3 diffuse = vec3(0.0, 0.85, 0.99) * lambda + (1. - lambda) * vec3(0.99, 0.25, 0.0);

    //diffuse =  vec3(0.41f, 0.05f, 0.675f) * nsteps/float(maxIters) + diffuse;

    // Normal for this position
    vec3 nor = calcNormal( pos );

    vec3 dir = getRay(camera, target, uv, fov);

    gl_FragColor = shading(pos, nor, dir, diffuse, nsteps)  ;

  }
}
