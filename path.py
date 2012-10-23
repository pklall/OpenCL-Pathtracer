#!/usr/bin/python

import numpy as np
import time

import numpy.linalg as la
import struct

import Image

import pyopencl as cl

# set width and height of window, more pixels take longer to calculate
w = 500
h = 500

class CLPathTracer():
	def __init__(self):
		# a dictionary of handles to opencl buffers we will allocate and use
		self.buffers = {}
		
		device = cl.get_platforms()[0].get_devices()
		self.ctx = cl.Context(device)
		print("Rendering on: " + str(device[0].name));
		self.queue = cl.CommandQueue(self.ctx)

	def prepare_program(self):
		f = file("pathtracer.cl", "r")
		self.prog = cl.Program(self.ctx, f.read()).build()
		f.close()
	
	def materialBuffer(self, materials):
		"""
		Creates a numpy array of material data provided data in the form:
		[(r, g, b, a, n, spec, ispec, emmit)]
		"""
		mat_array = []
		for (r, g, b, a, n, spec, ispec, emmit) in materials:
			mat_array.extend([r, g, b, a, n, spec, ispec, emmit])
		return np.array(mat_array, dtype=np.float32)

	def sceneA(self):
		"""
		Returns (cam, numSpheres, scene, materials, scene_material_id)
		"""
		mf = cl.mem_flags
		# position (f4), forward (f4), up (f4), right (f4),
		# heightAngle (f), aspectRatio (f)
		camera_numpy = struct.pack("ffffffffffffffffffii",\
				0, 0, -19, 1,\
				0, 0, 1, 1,\
				0, 1, 0, 1,\
				1, 0, 0, 1,\
				1.293, 1.0, w, h)
		camera_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,\
				hostbuf=camera_numpy)

		# 0 = light
		# 1 = light (yellow and less bright)
		# 2 = red surface
		# 3 = green surface
		# 4 = blue surface
		# 5 = white surface
		# 1 = light
		# 2 = matte red surface
		# 3 = shiny green surface
		# 4 = matte blue surface
		# 5 = mirror
		# 6 = frosted glass surface
		# 7 = clear glass surface
		materials_numpy = self.materialBuffer([\
				(1.0, 1.0, 1.0, 1.0, 0, 0, 0, 1), \
				(0.7, 0.7, 0.2, 1.0, 0, 0, 0, 1), \
				(1.0, 0.5, 0.5, 1.0, 1, 2, 0, 0), \
				(0.5, 1.0, 0.5, 1.0, 1, 2.0, 0, 0), \
				(0.5, 0.5, 1.0, 1.0, 1, 2, 0, 0), \
				(1.0, 1.0, 1.0, 1.0, 1, 2.0, 0, 0), \
				(1.0, 1.0, 1.0, 1.0, 2.0, 10.0, 0, 0), \
				(1.0, 1.0, 1.0, 0.0, 1.10, 2.0, 11, 0), \
				])
		materials_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,\
				hostbuf=materials_numpy)
		
		# a scene with 6 spheres for walls and small spheres inside
		# spheres are defined by:
		# x, y, z, radius
		walldist = 5000          #the distance from the origin to the center of each wall sphere
		wallrad = walldist - 20
		scene_numpy = np.array([\
				0, -26.5, -10, 15.0,\
				-15, -15, 15, 3.0,\
				0, 0, walldist, wallrad,\
				0, 0, -walldist, wallrad,\
				walldist, 0, 0, wallrad,\
				-walldist, 0, 0, wallrad,\
				0, walldist, 0, wallrad,\
				0, -walldist, 0, wallrad,\
				-5, 15, 8, 3,\
				0, -1, 10, 6,\
				8, 9, 3, 3,\
				10, 11, 10, 3,\
				10, -10, 10, 6,\
				8, 3, 2, 2\
			], dtype=np.float32)
		# assign materials to each sphere
		material_index_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,\
				hostbuf=np.array([\
				0, 1, 5, 2, 3, 4, 5, 2, 5, 6, 7, 2, 2, 0\
				], dtype=np.int32))
		scene_cl = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR,\
				hostbuf=scene_numpy)
		scene_size = struct.pack("i", 14)
		return (camera_cl, scene_size, scene_cl, materials_cl, material_index_cl)

	def render(self):
		"""
		Returns image data resulting from the render
		"""

		mf = cl.mem_flags
		# random number seeds
		random_cl = cl.Buffer(self.ctx, mf.READ_WRITE | mf.COPY_HOST_PTR,\
				hostbuf = np.random.bytes(w * h * 2 * 4))

		float_image_cl = cl.Buffer(self.ctx, mf.READ_WRITE, 4 * 4 * w * h);
		self.prog.initColorBuffer(self.queue, (w * h,), None, float_image_cl).wait();

		(camera, scene_size, scene, materials, material_index) = self.sceneA()

		numIterations = 200
		for i in range(numIterations):
			print("Pass: " + str(i) + " of " + str(numIterations))
			# NVidia cards supposedly don't refresh the display while a kernel is
			# operating, so we sleep to give the above print statement time to 
			# display and not lock the display of my single-gpu machine
			time.sleep(0.01)
			self.prog.pathtrace(self.queue, ((w * h), ), None,\
					random_cl,\
					camera,\
					scene_size,\
					scene,\
					materials,\
					material_index,\
					float_image_cl).wait()
		output_numpy = np.empty((w * h * 4,), dtype=np.uint8)
		output_cl = cl.Buffer(self.ctx, mf.WRITE_ONLY, output_numpy.nbytes)
		self.prog.getImage(self.queue, (w * h,), None, float_image_cl, output_cl);

		# read output
		cl.enqueue_read_buffer(self.queue, output_cl, output_numpy).wait()

		return output_numpy

if __name__ == '__main__':
	renderer = CLPathTracer()
	renderer.prepare_program()
	image_data = renderer.render()
	im = Image.frombuffer("RGBX", (w,h,), image_data, "raw", 'RGBX', 0, 1)
	im.show()
