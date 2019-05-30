import numpy
import time
from matplotlib import pyplot as plt
print ("start 10")

class Map:
	def __init__(self,gridwidth,sigmasignal):

		self.gridwidth = gridwidth

		self.griddiag = int(gridwidth*gridwidth*0.5)

		self.signalfreq = [[0]*gridwidth for i in range(gridwidth)]

		self.signalpixel = [[0]*gridwidth for i in range(gridwidth)]

		self.signalpixelwithphase = [[0]*gridwidth for i in range(gridwidth)]

		self.noiseymapwithphase = [[0]*gridwidth for i in range(gridwidth)]
		
		self.noiseymap =  [[0]*gridwidth for i in range (gridwidth)]

		self.sigmasignal = sigmasignal

		self.PowerSpectrum = [0 for i in range(int(gridwidth*gridwidth*0.5))] # PowerSpectrum of signal

		self.NoiseCovariance = [0 for i in range(gridwidth*gridwidth)]

		print("created new map of size: ", gridwidth)

	def PowerSpectrumGenerator(self, map): #reconstructs the power spectrum of the reconstructed map
		##############################################
		#Generate Bins
		#############################################
		binsize = 1
		Bins = [int(i/binsize) for i in range(int(gridwidth*gridwidth/2))]
		countk=[0 for i in range(int(gridwidth*gridwidth*0.5))]
		magkone =[0 for i in range(int(gridwidth*gridwidth*0.5))] # magnitude of the wave vector
		ReconPowerSpectrum = [0 for i in range(int(gridwidth*gridwidth*0.5))] # Reconstructed PowerSpectrum of signal
		n = 0
		m = 0
		while n < gridwidth:
			while m < gridwidth:
				k = (float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
				k=k-1 #shifts k forwards for easy maths
				k=int(k)
				k = Bins[k]
				ReconPowerSpectrum[k] = (ReconPowerSpectrum[k]*countk[k]+numpy.abs(map[n][m]*pixelwidth))/(countk[k]+1)
				countk[k] = countk[k]+1
				m = m + 1
			n = n + 1
			m = 0

		i=0
		while i < int(gridwidth*gridwidth*0.5): #create powerspectrum list for values along diagonal in k space
			magkone[i]=i
			i=i+1

		ReconPowerSpectrum.insert(0, 0) #shifts k back so that the k=0 => S(k)=0
		L=len(ReconPowerSpectrum)-1
		ReconPowerSpectrum.pop(L)
		countk.insert(0, 0) #defines the 0 frequency mode
		L=len(countk)-1
		countk.pop(L)

		i=1
		while i < len(countk): #delete empty values from the spectrum
			if countk[i] == 0:
				countk.pop(i)
				ReconPowerSpectrum.pop(i)
				magkone.pop(i)
				i=i-1
			i=i+1
		
		newishlist=[numpy.sqrt(x) for x in magkone]
		magkone=newishlist
		return magkone, ReconPowerSpectrum #return k and Powerspectrum(k)

	####################################
	# define the required power spectrum
	####################################
	def powerspectrumfunc(self, n): #defines the power spectrum at the nth diag in k space
		sigmasignal = self.sigmasignal
		function=numpy.exp(-float(n)*float(n)/(2*sigmasignal*sigmasignal))/(numpy.sqrt(2*numpy.pi)*(sigmasignal))
		# this function is in fact the square of the pwoer spectrum
		if(function < 0):
			print ("Error, please check inputs")
		if(function ==0):
			function = 5.03739653754e-316
		return function

	def phasefactor(self, input): #solves the (-1)^n phase factor input
		gridwidth = self.gridwidth
		n=0
		m=0
		while n < gridwidth:
			while m < gridwidth:
				input[n][m] = input[n][m]*numpy.power(-1,n+m)
				m=m+1
			n=n+1
			m=0
		return input

	def WienerMessengerField(self, N,Sft,datavector,iterations): #Function that returns the reconstructed map
		print ("Started Messenger Field Function")
		tau=0.9 #note lambda = 1.0 implicitely
		t = [0 for i in range(gridwidth*gridwidth)]
		s = [0 for i in range(gridwidth*gridwidth)]
		twoDt = [[0]*gridwidth for i in range(gridwidth)] #noise covariance
		twoDs = [[0]*gridwidth for i in range(gridwidth)] #signal covariance
		Nbarinv=[0 for i in range (gridwidth*gridwidth)]
		T=[sigmanoise*sigmanoise*tau for i in range (gridwidth*gridwidth)]
		Tinv=[0 for i in range (gridwidth*gridwidth)]
		Tftinv=[0 for i in range (gridwidth*gridwidth)]
		Sftinv=[0for i in range (gridwidth*gridwidth)]
		RHS=[0 for i in range (gridwidth*gridwidth)]
		RHS2=[0 for i in range (gridwidth*gridwidth)]
		i=0
		max=len(Nbarinv)

		while i <max: #generate the Nbar=N-T and T in the Fourier domain
			a=float(N[i])
			b=float(T[i])
			c=float(Sft[i])
			Nbarinv[i]=float(float(1)/(a-b))
			Tinv[i]=float(float(1)/b)
			Tftinv[i]=b*(gridwidth*gridwidth)
			Sftinv[i] = float(float(1)/c)
			i=i+1

		k = 0
		while k < iterations:

			start = time.clock()

			i = 0
			while i < max: #calculate t in the pixel domain
				RHS[i] = Nbarinv[i]*datavector[i] + Tinv[i]*s[i]
				t[i] = RHS[i]*(1.0/(Nbarinv[i]+Tinv[i]))
				i = i + 1

			i = 0
			j= 0
			while i < gridwidth: #wrap t into a matrix
				while j < gridwidth:
					twoDt[i][j]= t[i*gridwidth+j]
					j = j + 1
				i = i + 1
				j = 0

			middle1 = time.clock() #this function gives timing for each iteration of the messenger field
			print ("For iteration ", i, " Dot Product took ", middle1-start, " seconds")
			twoDt = numpy.fft.fft2(twoDt) #Fourier transform t
			middle2 = time.clock()
			print ("For iteration ", i, " FT took ", middle2-middle1, " seconds")
			i = 0
			j= 0
			while i < gridwidth: #unwrap FT(t)
				while j < gridwidth:
					t[i*gridwidth+j]= twoDt[i][j]
					j = j + 1
				i = i + 1
				j = 0

			i = 0
			while i < max: #calculate s int he Fourier domain
				RHS2[i] = Tftinv[i]*t[i]
				s[i] = RHS2[i]*(1.0/(Sftinv[i]+Tftinv[i]))
				i = i + 1

			i = 0
			j= 0
			while i < gridwidth:
				while j < gridwidth:
					twoDs[i][j]= s[i*gridwidth+j]
					j = j + 1
				i = i + 1
				j = 0

			middle3 = time.clock()
			
			print ("For iteration ", i, " Dot Product 2 took ", middle3-middle2, " seconds")
			twoDs = numpy.fft.ifft2(twoDs)
			middle4 = time.clock()
			print ("For iteration ", i, " FT2 Took ", middle4 - middle3, " seconds")

			i = 0
			j= 0
			while i < gridwidth:
				while j < gridwidth:
					s[i*gridwidth+j]= twoDs[i][j]
					j = j + 1
				i = i + 1
				j = 0

			print (str(k) + " done")
			k = k + 1
			print middle1-start, middle2-middle1, middle3-middle2,middle4 - middle3

		return s

	def generateMap(self):
		gridwidth = self.gridwidth

		n=1
		m=1
		while m < gridwidth/2: #create array for Re(s_k) + sqrt(-1) Im(s_k): <s_k s_kprime> = S(k) deltakkprime
			while n<gridwidth: #each imaginary and real part sampled from N(0,S(k))
				k = numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
				im=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*self.powerspectrumfunc(k), size =None)
				re=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*self.powerspectrumfunc(k), size =None)
				self.signalfreq[n][m]=re+1j*im
				self.signalfreq[-n][-m]=re-1j*im
				n=n+1
			n=1
			m=m+1
		#########################################
		# generate all centre column entries (except 0,0 which can be defined as a sample from N(0,0), which is a delta at 0)
		#############################################
		n=1
		m=int(gridwidth/2)
		while n<gridwidth/2:
			k = numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
			im=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*self.powerspectrumfunc(k), size = None)
			re=numpy.random.normal(loc=0.0,scale = numpy.sqrt(0.5)*self.powerspectrumfunc(k), size = None)
			self.signalfreq[n][m]=re+1j*im
			self.signalfreq[-n][-m]=re-1j*im
			n=n+1
		########################################################
		# generate the edge entries as real samples from N(0,2*S(k))
		########################################################
		n=0
		m=0
		while n<gridwidth:
			re=numpy.random.normal(loc=0.0,scale =self.powerspectrumfunc(numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))), size = None)
			self.signalfreq[n][m]=re
			n=n+1
		m=1
		n=0
		
		while m<gridwidth:
			re=numpy.random.normal(loc=0.0,scale = self.powerspectrumfunc(numpy.sqrt(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))), size = None)
			self.signalfreq[n][m]=re
			m=m+1

		self.signalfreq[int(gridwidth/2)][int(gridwidth/2)] = self.powerspectrumfunc(0)

		self.signalfreq = numpy.array(self.signalfreq)
		self.signalpixel = numpy.real(numpy.fft.fft2(self.signalfreq))
		self.signalpixelwithphase = self.phasefactor(self.signalpixel+0.0)

		# return self.signalfreq #generate ft into pixel space

	def addNoise(self, sigmanoise):
		gridwidth = self.gridwidth
		n=0
		m=0
		while n<gridwidth: #form a data vector (plus some others)
			while m < gridwidth:
				noise=numpy.random.normal(loc=0.0, scale = sigmanoise, size = None)
				self.NoiseCovariance[n*gridwidth+m] = sigmanoise*sigmanoise
				self.noiseymap[n][m]=self.signalpixel[n][m] + noise
				self.noiseymapwithphase[n][m]=self.signalpixelwithphase[n][m]+ noise
				m=m+1
			m=0
			n=n+1
		self.noiseymap = numpy.array(self.noiseymap)
		self.NoiseCovariance = numpy.array(self.NoiseCovariance)


	def generatePowerspectrum(self):
		magk = [0 for i in range(self.griddiag)]
		i=0
		while i < self.griddiag: #create powerspectrum list for values along diagonal in k space
			magk[i]=i
			self.PowerSpectrum[i] = self.powerspectrumfunc(i)
			i=i+1

		return magk, self.PowerSpectrum

	def saveMap(self):
		outfile = 'mapi64'
		outfiletwo = 'truemapi64'
		numpy.savetxt(outfile, self.noiseymapwithphase, delimiter=',')
		numpy.savetxt(outfiletwo, self.signalpixelwithphase, delimiter=',')


