import numpy as np
import torch

class PLL():
	def __init__(self,
		filter_state = 0, vco_free = 50*2*np.pi, amplitude = 1, K_vco = 100):
		self.filter_state = filter_state
		self.vco_freq = vco_free  # in Hz
		self.vco_phase = 0
		self.K_vco = K_vco
		self.amplitude = amplitude

	def forward_VCO(self, control, duration):
		self.vco_freq = self.vco_freq + self.K_vco * control * duration
		self.vco_phase = self.vco_phase + (2*self.vco_freq - self.K_vco * control * duration)/2*duration
		self.vco_phase = self.vco_phase % (2*np.pi)
		out = self.amplitude * torch.sin(self.vco_phase)		
		return out

	def detach(self):		
		self.vco_freq = self.vco_freq.detach()
		self.vco_phase = self.vco_phase.detach()		
