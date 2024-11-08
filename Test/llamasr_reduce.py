from Trace.traceLlamas import TraceLlamas
from Extract.extractLlamas import ExtractLlamas
# from Arc.arcLlamas import ArcLlamas

flatfile = 'FinalAlignmentData/Bench_1a_Final_Alignment_Data/Red/REF_fIFU_Flat/short/20240506_151435_Camera4_1A_Red_0.05s-2.0e_Flat_Short.fits'

trace  = TraceLlamas(flatfile,spectrograph=0,channel='red',mph=1500)
trace.profileFit()
# arcsol = ArcLlamas("Green_Kr.fits",trace)
