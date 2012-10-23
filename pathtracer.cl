#pragma OPENCL EXTENSION cl_khr_byte_addressable_store : enable

typedef struct Camera{
	float4 position;
	float4 direction;
	float4 up;
	float4 right;
	float heightAngle;
	float aspectRatio;
	int width;
	int height;
} Camera;

typedef struct Ray {
	float4 position;  // last coordinate ignored
	float4 direction; // last coordinate ignored
} Ray;

typedef struct Sphere {
	float4 center; // (x, y, z) = center, w = radius
} Sphere;

typedef struct Material {
	float4 color; // rgba
	float n;      // index of refraction (1 = air)
	float spec;   // specularity (0 = matte, 1 = shiny, >1 = shinier)
	float ispec;  // refractive specularity (0 = frosted glass, 1 = a little frosted, >1 = sharp/clear refraction)
	float emmit;  // emitance (1 = light source, 0 = regular surface)
} Material;

typedef struct IInfo {
	float4 normal;
	float4 position;
	float distance;
} IInfo;

// Holds state for random number generation
typedef struct Rand {
	int2 r;
} Rand;

// multiply-with-carry pseudo-random number generator
float nextRand(Rand* rand) {
	// rand->r = ((int2) (36969, 18000)) * ((rand->r & 65535) + (rand->r >> 16));
	rand->r.x = 36969  * ((rand->r.x) & 65535) + ((rand->r.x) >> 16);
	rand->r.y = 18000 * ((rand->r.y) & 65535) + ((rand->r.y) >> 16);
	
	// random 32-bit value
	uint r = ((rand->r.x) << 16) + (rand->r.y);
	// return a float between 0 and 1
	int intpart;
	return (frexp(fabs(as_float(r)), &intpart) - 0.5f) * 2.0f;
}


Ray genray(__constant struct Camera* cam, const int gid, Rand* rand) {
	// the coordinates of the current ray's pixel
	float2 ij = (float2) ((float) (gid % cam->width), (float) (gid / cam->height));
	float2 size = (float2) ((float) cam->width, (float) cam->height);
	
	Ray ray; // the ray to return


	// All of the following assumes the distance to the projection plane is 1

	// scalars such that cam->ra->up * maxScaleU is the top edge 
	// of the projection plane, and cam->ra->right * maxScaleR is the right edge 
	// of the projection plane
	float maxScaleU = tan(cam->heightAngle/2.0);
	float maxScaleR = cam->aspectRatio * maxScaleU;
	// coordinates of the pixel for this ray normalized to [0, 1]
	// JITTERED SAMPLING:
	float xjitter = nextRand(rand);
	float yjitter = nextRand(rand);
	float2 scaleRU = (ij + (float2) (xjitter, yjitter)) / size;

	ray.direction = cam->direction +
		(cam->right * (2.0 * scaleRU.x - 1.0) * maxScaleR) +
		(cam->up * (2.0 * scaleRU.y - 1.0) * maxScaleU);
	ray.position = cam->position;
	ray.position.w = 0;
	ray.direction.w = 0;
	ray.direction = normalize(ray.direction);

	return ray;
}

IInfo raySphereIntersection(Ray ray, float maxT, Sphere sphere) {
	IInfo iInfo;
	float4 center = sphere.center;
	float radius = center.w;
	center.w = 0;
	ray.position.w = 0;
	ray.direction.w = 0;

	float4 rp = ray.position - center;
	float a = 1.0f;
	float b = 2*dot(rp, ray.direction);
	float c = dot(rp, rp) - radius * radius;
	float descrim = b*b - 4*a*c;
	if(descrim < 0.0) {
		iInfo.distance = -1;
		return iInfo;
	}

	float t;
	float outside; // 1 if the ray starts outside the sphere, -1 otherwise
	// if the ray starts outside the sphere
	if(c > 0) {
		outside = 1;
	} else {
		outside = -1;
	}
	t = (-b - outside * sqrt(descrim)) / (2.0 * a);
	iInfo.distance = t;
	iInfo.position = ray.direction * t + ray.position;
	iInfo.normal = normalize(iInfo.position - center);
	return iInfo;
}

/*
	Generates a random ray based on the hemisphere around the surface normal.
*/
Ray getNextRay(IInfo iInfo, Rand* r) {
	Ray ray;
	ray.position = iInfo.position;
	// get random spherical coordinates 
	// normalize the random numbers
	float rnd1 = nextRand(r);
	float rnd2 = nextRand(r);
	// we must transform by acos so we don't get clustering at the poles
	float u = 2 * rnd1 - 1;
	float v = 2 * M_PI * rnd2;
	float x = sqrt(1 - u * u) * cos(v);
	float y  = sqrt(1 - u * u) * sin(v);
	float z = u;
	ray.direction = (float4) (x, y, z, 0);
	ray.direction = normalize(ray.direction);
	// flip ray direction so we're inside (relative to the normal)
	if(dot(ray.direction, iInfo.normal) < 0) {
		ray.direction *= -1;
	}
	return ray;
}

float4 reflect(float4 v, float4 n){
	// assumes that n is already normalized
	float4 proj = n * dot(v, n);
	return proj * 2.0 - v;
}

float4 refract(float4 v, float4 normal, float n){
	// assumes that n is already normalized
	float ct1 = dot(normal, -1 * v);
	float ct2 = sqrt(1 - n*n*(1 - ct1*ct1));
	return n * v + (n * ct1 - ct2) * normal;
}

__kernel void pathtrace(
	// seeds for the random number generator (2 ints required per kernel call)
	__global Rand * randomSeed,
	// camera
	__constant struct Camera* cam,
	// the number of elements in the scene
	const int numSpheres,
	// list of scene elements (only spheres for now)
	__global const Sphere * scene,
	// list of materials
	__global const Material * materials,
	// mapping from the scene element list to materials 
	// materials[scene_material[i]] is the material associated with scene[i]
	__global const int * scene_material_id,
	// output color buffer
	__global float4 * color) {

	float minPathContrib = 0.000001;

	int gid = get_global_id(0);

	Rand rand = randomSeed[gid];

	// the total accumulation of light from all paths
	float4 totalColor = (float4) (0,0,0,0);
	// the accumulated color of the current path
	float4 pathColor = (float4) (1,1,1,0);
	// the number of paths found
	int numPaths = 0;
	const int maxPathLength = 10;
	int curPathLength = 0;
	Ray ray = genray(cam, gid, &rand);
	for(int numRayCasts = 100; numRayCasts > 0; numRayCasts--) {
		curPathLength++;
		IInfo iInfo;
		iInfo.distance = MAXFLOAT;
		int matId = -1;
		// move the ray slightly to avoid intersecting the same geometry
		ray.position += ray.direction * 0.01;
		// get the nearest intersection by checking against all spheres
		for(int i = 0; i < numSpheres; i++) {
			IInfo current = raySphereIntersection(ray, iInfo.distance, scene[i]);
			if(current.distance > 0 && current.distance < iInfo.distance) {
				iInfo = current;
				matId = i;
			}
		}
		Material mat = materials[scene_material_id[matId]];
		// if we hit something
		if(matId != -1) {
			// if we hit a light, add the path color
			if(mat.emmit > 0) {
				totalColor += pathColor * mat.color;
			} else{ // if we hit a regular object
				bool insideCollision = dot(ray.direction, iInfo.normal) > 0;
				// compute the next ray
				Ray next = getNextRay(iInfo, &rand);
				if(insideCollision) {
					next.direction *= -1;
				}
				float4 expectedDir;
				float spec;
				float lambert = 1.0;
				// use alpha as probability of going through the object
				if(nextRand(&rand) > mat.color.w) {
					// if it is refracted
					next.direction *= -1;
					float n; // index of refraction relative to air
					// if we are going into the solid
					if(dot(iInfo.normal, ray.direction) > 0) {
						n = 1.0f / mat.n;
						iInfo.normal *= -1;
					} else {
						n = mat.n;
					}
					expectedDir = normalize(refract(ray.direction, iInfo.normal, n));
					spec = mat.ispec;
				} else {
					// if it is reflected
					expectedDir = normalize(reflect(ray.direction, iInfo.normal));
					spec = mat.spec;
					lambert = dot(next.direction, iInfo.normal);
				}
				// ugly hack:
				// if the object is extremely specular, cast rays where they should go only
				if(spec > 10) {
					next.direction = expectedDir;
				}
				// percentage of reflected light (based on specularity)
				float brdf = clamp(pow(dot(next.direction, expectedDir) + 0.1, spec), 0.0f, 1.0f);
				pathColor *= (float4) mat.color * brdf * lambert;
				ray = next;
			}
		}
		// if we hit a light, or nothing at all
		if(mat.emmit > 0 || matId == -1 || curPathLength > maxPathLength || (
					pathColor.x < minPathContrib && pathColor.y < minPathContrib && pathColor.z < minPathContrib)) {
			// reset the ray
			ray = genray(cam, gid, &rand);
			pathColor = (float4) (1,1,1,0);
			curPathLength = 0;
			numPaths++;
		}
	}

	totalColor /= numPaths;

	float4 currentColor = color[gid];
	color[gid] = totalColor + currentColor;
	// update the random seed
	randomSeed[gid] = rand;
}

// initialize the color buffer with all 0's
__kernel void initColorBuffer(__global float4 * color) {
	int gid = get_global_id(0);
	color[gid] = (float4) (0,0,0,0);
}

// convert the float color buffer into a uchar image buffer we can use
// max defines the maximum color value used for normalization 
__kernel void getImage(
	__global const float4 * color, __global const uchar4 * output) {
	int gid = get_global_id(0);
	// crude tone mapping
	float3 c = (color[gid]).xyz;
	float3 tone = c / (c + (float3) (1, 1, 1));
	output[gid].x = (uchar) (clamp(tone.x, 0.0f, 1.0f) * 255);
	output[gid].y = (uchar) (clamp(tone.y, 0.0f, 1.0f) * 255);
	output[gid].z = (uchar) (clamp(tone.z, 0.0f, 1.0f) * 255);
	output[gid].w = 255;
}
