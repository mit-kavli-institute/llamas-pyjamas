from LlamasPipeline.Trace.traceLlamas import TraceLlamas
from LlamasPipeline.Extract.extractLlamas import ExtractLlamas
from LlamasPipeline.Arc.arcLlamas import ArcLlamas

flatfile = 'FinalAlignmentData/Bench_1a_Final_Alignment_Data/Blue/REF_fIFU_Flat/short/20240506_151730_Camera2_1A_Blue_0.4s-2.0e_Flat_Short.fits'

trace  = TraceLlamas(flatfile,spectrograph=0,channel='blue',mph=1500)
trace.profileFit()

#arcsol = ArcLlamas("20200110_134641_Camera00_blue_KrNeAr_0.2s_BASELINE.fits",trace)
#arcsol = ArcLlamas("20200110_134705_Camera00_blue_KrNeAr_4.0s_BASELINE.fits",trace)


