import numpy
import time
from matplotlib import pyplot as plt
from mapclass import Map

#########
# define rewrap matrix
##############
def wrap(v):
	length = len(v)
	sqrtlength = int(numpy.power(length,0.5))
	twoDt=[[0]*sqrtlength for i in range (sqrtlength)]
	i = 0
	j= 0
	while i < sqrtlength:
		while j < sqrtlength:
			twoDt[i][j]= v[i*sqrtlength+j]
			j = j + 1
		i = i + 1
		j = 0
	twoDt=numpy.array(twoDt)
	return twoDt

############
# define unwrap mamtrix
###############
def unwrap(M):
	M = numpy.array(M)
	rows = M.shape[0]
	columns = M.shape[1]
	t=[0]*rows*columns
	i = 0
	j= 0
	while i < rows:
		while j < columns:
			t[i*rows+j]= M[i][j]
			j = j + 1
		i = i + 1
		j = 0
	return t

def PowerSpectrumGenerator(map): #reconstructs the power spectrum of the reconstructed map
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
			k=int(k-1) #shifts k forwards for easy maths
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
def powerspectrumfunc(n): #defines the power spectrum at the nth diag in k space
	sigmasignal = 3.0
	function=numpy.exp(-float(n)*float(n)/(2*sigmasignal*sigmasignal))/(numpy.sqrt(2*numpy.pi)*(sigmasignal))
	# this function is in fact the square of the pwoer spectrum
	if(function < 0):
		print ("PANIC")
	if(function ==0):
		function = 5.03739653754e-316
	return function

def phasefactor(input): #solves the (-1)^n phase factor input
	n=0
	m=0
	while n < gridwidth:
		while m < gridwidth:
			input[n][m] = input[n][m]*numpy.power(-1,n+m)
			m=m+1
		n=n+1
		m=0
	return input

def WienerMessengerField(N,Sft,datavector,iterations): #Function that returns the reconstructed map

	print ("Started Messenger Field Function")
	tau=0.9 #note lambda = 1.0 implicitely
	t = [0 for i in range(gridwidth*gridwidth)]
	s = [0 for i in range(gridwidth*gridwidth)]
	T=[sigmanoise*sigmanoise*tau for i in range (gridwidth*gridwidth)]
	T = numpy.array(T)
	Sft = numpy.array(Sft)

	Nbarinv = 1.0/(N-T)
	
	print('sigmanoise: ',sigmanoise)
	Tinv=1.0/T
	Tftinv = T*gridwidth*gridwidth
	Sftinv = 1.0/Sft

	RHS=[0 for i in range (gridwidth*gridwidth)]
	RHS2=[0 for i in range (gridwidth*gridwidth)]

	max=len(Nbarinv)

	k = 0
	while k < iterations:

		start = time.clock()

		i = 0
		while i < max: #calculate t in the pixel domain
			RHS[i] = Nbarinv[i]*datavector[i] + Tinv[i]*s[i]
			t[i] = RHS[i]*(1.0/(Nbarinv[i]+Tinv[i]))
			i = i + 1

		twoDt = wrap(t)

		middle1 = time.clock() #this function gives timing for each iteration of the messenger field
		print ("For iteration ", i, " Dot Product took ", middle1-start, " seconds")
		twoDt = numpy.fft.fft2(twoDt) #Fourier transform t
		middle2 = time.clock()
		print ("For iteration ", i, " FT took ", middle2-middle1, " seconds")
		t = unwrap(twoDt)

		i = 0
		while i < max: #calculate s int he Fourier domain
			RHS2[i] = Tftinv[i]*t[i]
			s[i] = RHS2[i]*(1.0/(Sftinv[i]+Tftinv[i]))
			i = i + 1

		twoDs = wrap(s)

		middle3 = time.clock()
		
		print ("For iteration ", i, " Dot Product 2 took ", middle3-middle2, " seconds")
		twoDs = numpy.fft.ifft2(twoDs)
		middle4 = time.clock()
		print ("For iteration ", i, " FT2 Took ", middle4 - middle3, " seconds")

		s = unwrap(twoDs)

		print (str(k) + " done")
		k = k + 1
		print middle1-start, middle2-middle1, middle3-middle2,middle4 - middle3

	return s

#######################################
#define the arrays and lengths
#######################################
if __name__ == '__main__':
	pixelwidth = 1.0 # width and length of each pixel
	global gridwidth
	gridwidth=64# make 2^n for int n: this is the edge length of the map
	sigmanoise=0.3# noise variance
	griddiag = int(gridwidth*gridwidth*0.5)
	map = [[0]*gridwidth for i in range(gridwidth)] #Generated data map
	Wiener= [[0]*gridwidth for i in range(gridwidth)]#Wienered map using standard function
	# powermap= [[0]*gridwidth for i in range(gridwidth)]#powerspectrum in 2D map
	magk =[0 for i in range(griddiag)] # magnitude of the wave vector
	PowerSpectrum =[0 for i in range(griddiag)] # PowerSpectrum of signal
	sigmasignal = 3.0
	countk=[0 for i in range(int(gridwidth*gridwidth*0.5))]
	ReconPowerSpectrum = [0 for i in range(int(gridwidth*gridwidth*0.5))] # Reconstructed PowerSpectrum of signal
	xaxis = [0 for i in range(gridwidth*gridwidth)]
	yaxis = [0 for i in range(gridwidth*gridwidth)]
	yaxis2 = [0 for i in range(gridwidth*gridwidth)]
	print ("DONE INITIALISING 10")
	mapobj = Map(gridwidth,sigmasignal)

	magk, PowerSpectrum = mapobj.generatePowerspectrum()
	plt.plot(magk,PowerSpectrum)

	# plt.show()
	##############################################
	#Generate Bins for the reconstructed power spectrum
	#############################################
	binsize = 1
	Bins = [int(i/binsize) for i in range(int(gridwidth*gridwidth/2))]
	###################################################################
	# generate all left of centre entries to signal frequency matrix and their conjugate symmetry entries
	#######################################################
	repeats = 0
	while repeats < 1:
		reconvector=[]
		
		mapobj.generateMap()
		signalfreq = mapobj.signalfreq
		signalpixel2 = mapobj.signalpixelwithphase

		mapobj.addNoise(sigmanoise)

		datavector = unwrap(mapobj.noiseymap)

		filteredmap = [[0]*gridwidth for i in range(gridwidth)]
		##################
		## Forming N cov matrix
		#################################
		N = mapobj.NoiseCovariance

		Sft = [0 for i in range(gridwidth*gridwidth)]
		#####################
		## Forming Sft convariance
		########################
		n = 0
		m = 0
		while m < gridwidth:
			while n < gridwidth:
				k = numpy.power((n-gridwidth/2)*(n-gridwidth/2) + (m-gridwidth/2)*(m-gridwidth/2),0.5)
				Sft[m*gridwidth + n] = powerspectrumfunc(k)
				n = n + 1
			m = m + 1
			n = 0

		reconvector = WienerMessengerField(N,Sft,datavector,30) #calculate the reconstructed map with 10 iterations
		n = 0
		m = 0
		while n < gridwidth:
			while m < gridwidth:
				filteredmap[n][m] = float(numpy.real(reconvector[n*gridwidth+m]))+1j*float(numpy.imag(reconvector[n*gridwidth+m]))
				m = m + 1
			m=0
			n = n+ 1
		#####################################
		#Reconstruct Power Spectrum
		#####################################
		FTReconmap = numpy.fft.fft2(filteredmap)/(gridwidth*gridwidth) # Fourier of Reconstructedmap
		n = 0
		m = 0
		counting = 0
		while n < gridwidth:
			while m < gridwidth:
				k =(float(n-(gridwidth/2))*float(n-(gridwidth/2))+float(m-(gridwidth/2))*float(m-(gridwidth/2)))
				k=k-1 #shifts k forwards for easy maths
				k=int(k)
				k = Bins[k]
				ReconPowerSpectrum[k] = (ReconPowerSpectrum[k]*countk[k]+numpy.abs(FTReconmap[n][m]*pixelwidth))/(countk[k]+1)
				countk[k] = countk[k]+1
				m = m + 1
			n = n + 1
			m = 0
		repeats = repeats+1

	kval= [0 for i in range(int(gridwidth*gridwidth*0.5))] #define same as magk
	magkone =[0 for i in range(int(gridwidth*gridwidth*0.5))] # magnitude of the wave vector
	i=0
	while i < int(gridwidth*gridwidth*0.5): #create powerspectrum list for values along diagonal in k space
		magkone[i]=i
		i=i+1

	ReconPowerSpectrum.insert(0, 0) #shifts k back so that the k=0 => S(k)=0
	L=len(ReconPowerSpectrum)-1
	ReconPowerSpectrum.pop(L)
	countk.insert(0, 0)
	L=len(countk)-1
	countk.pop(L)

	i=1
	while i < len(countk): #delete empty values from
		if countk[i] == 0:
			countk.pop(i)
			ReconPowerSpectrum.pop(i)
			magkone.pop(i)
			i=0	
		i=i+1

	newishlist=[numpy.sqrt(x) for x in magkone]
	magkone=newishlist

	i = int(gridwidth*numpy.sqrt(0.5))

	while i < griddiag:
		PowerSpectrum.pop(int(gridwidth*numpy.sqrt(0.5)))
		magk.pop(int(gridwidth*numpy.sqrt(0.5)))
		i = i + 1

	magk2, ReconPowerSpectrum2=PowerSpectrumGenerator(signalfreq)
	x=numpy.log(-numpy.array(ReconPowerSpectrum)+numpy.array(ReconPowerSpectrum2))
	y=numpy.log(numpy.array(ReconPowerSpectrum))
	plt.subplot(1,3,1),plt.plot(magk,numpy.log(PowerSpectrum))
	plt.subplot(1,3,2),plt.scatter(magkone,y)
	plt.subplot(1,3,3),plt.scatter(magk2,x)
	# plt.show()
	# plt.savefig("ReconMessengerField")

	###########################################
	# plot the maps
	###########################################
	plt.subplot(2,2,1),plt.imshow(numpy.real(mapobj.signalpixelwithphase), cmap = "jet")
	plt.ylabel("True Map")
	plt.colorbar()
	plt.subplot(2,2,2),plt.imshow(numpy.real(mapobj.noiseymapwithphase), cmap = "jet")
	plt.ylabel("Map + Noise")
	plt.colorbar()
	plt.subplot(2,2,3),plt.imshow(numpy.real(phasefactor(filteredmap)), cmap = "jet")
	plt.ylabel("Filtered Map")
	plt.colorbar()
	plt.subplot(2,2,4),plt.imshow(numpy.real(signalpixel2-(filteredmap)), cmap = "jet")
	plt.ylabel("Difference")
	plt.colorbar()
	plt.show()

	print ("done")
