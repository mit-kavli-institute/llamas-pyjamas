from LlamasPipeline.Trace.traceLlamas import TraceLlamas
from LlamasPipeline.Extract.extractLlamas import ExtractLlamas
from LlamasPipeline.Arc.arcLlamas import ArcLlamas

dotrace   = True
doextract = True
doarc     = False

flatfile1 = 'FinalAlignmentData/Bench_1a_Final_Alignment_Data/Green/REF_fIFU_Flat/short/20240506_151602_Camera3_1A_Green_0.1s-2.0e_Flat_Short.fits'
flatfile2 = 'FinalAlignmentData/Bench_2a_Final_Alignment_Data/Green/REF_fIFU_Flat/short/20240510_145415_Camera6_2A_Green_0.1s-2.0e.fits'
flatfile3 = 'FinalAlignmentData/Bench_3a_Final_Alignment_Data/3A-Green/REF_fIFU_Flat/short/3A-G_fIFU_Flat_0.1s-2.0e__20240423_133251.fits'
flatfile4 = 'FinalAlignmentData/Bench_4a_Final_Alignment_Data/Green/REF_FIFU_Flat/short/20240617_144205_Camera12_4A_Green_0.1s-2.0e.fits'
flatfile1b = 'FinalAlignmentData/Bench_1b_Final_Alignment_Data/Green/REF_fIFU_Flat/short/20240815_072823_Camera23_1B_Green_0.1s-2.0e_short.fits'
flatfile2b = 'FinalAlignmentData/Bench_2b_Final_Alignment_Data/Green/REF_fIFU_Flat/short/20240815_072722_Camera11_2B_Green_0.1s-2.0e_short.fits'
flatfile3b = 'FinalAlignmentData/Bench_3b_Final_Alignment_Data/Green/REF_fIFU_Flat/short/20240815_072837_Camera17_3B_Green_0.1s-2.0e_short.fits'
flatfile4b = 'FinalAlignmentData/Bench_4b_Final_Alignment_Data/Green/REF_fIFU_Flat/short/20240815_072844_Camera25_4B_Green_0.1s-2.0e_short.fits'


if (dotrace):
    """
    trace1  = TraceLlamas(flatfile1,spectrograph='1A',channel='green',mph=5000)
    trace1.profileFit()
    trace1.saveTrace(outfile='Trace_1A.pkl')
    trace2  = TraceLlamas(flatfile2,spectrograph='2A',channel='green',mph=5000)
    trace2.profileFit()
    trace2.saveTrace(outfile='Trace_2A.pkl')
    trace3  = TraceLlamas(flatfile3,spectrograph='3A',channel='green',mph=5000)
    trace3.profileFit()
    trace3.saveTrace(outfile='Trace_3A.pkl')
    trace4  = TraceLlamas(flatfile4,spectrograph='4A',channel='green',mph=5000)
    trace4.profileFit()
    trace4.saveTrace(outfile='Trace_4A.pkl')
    trace1b  = TraceLlamas(flatfile1b,spectrograph='1B',channel='green',mph=5000)
    trace1b.profileFit()
    trace1b.saveTrace(outfile='Trace_1B.pkl')
    """
    trace2b  = TraceLlamas(flatfile2b,spectrograph='2B',channel='green',mph=5000)
    trace2b.profileFit()
    trace2b.saveTrace(outfile='Trace_2B.pkl')
    """
    trace3b  = TraceLlamas(flatfile3b,spectrograph='3B',channel='green',mph=5000)
    trace3b.profileFit()
    trace3b.saveTrace(outfile='Trace_3B.pkl')
    trace4b  = TraceLlamas(flatfile4b,spectrograph='4B',channel='green',mph=5000)
    trace4b.profileFit()
    trace4b.saveTrace(outfile='Trace_4B.pkl')
    """
    
if (doextract):
    extraction1 = ExtractLlamas(flatfile1, TraceLlamas.loadTrace('Trace_1A.pkl'))
    extraction2 = ExtractLlamas(flatfile2, TraceLlamas.loadTrace('Trace_2A.pkl'))
    extraction3 = ExtractLlamas(flatfile3, TraceLlamas.loadTrace('Trace_3A.pkl'))
    extraction4 = ExtractLlamas(flatfile4, TraceLlamas.loadTrace('Trace_4A.pkl'))
    extraction1.saveExtraction(outfile='Extract_1A.pkl')
    extraction2.saveExtraction(outfile='Extract_2A.pkl')
    extraction3.saveExtraction(outfile='Extract_3A.pkl')
    extraction4.saveExtraction(outfile='Extract_4A.pkl')
    extraction1b = ExtractLlamas(flatfile1b, TraceLlamas.loadTrace('Trace_1B.pkl'))
    extraction2b = ExtractLlamas(flatfile2b, TraceLlamas.loadTrace('Trace_2B.pkl'))
    extraction3b = ExtractLlamas(flatfile3b, TraceLlamas.loadTrace('Trace_3B.pkl'))
    extraction4b = ExtractLlamas(flatfile4b, TraceLlamas.loadTrace('Trace_4B.pkl'))
    extraction1b.saveExtraction(outfile='Extract_1B.pkl')
    extraction2b.saveExtraction(outfile='Extract_2B.pkl')
    extraction3b.saveExtraction(outfile='Extract_3B.pkl')
    extraction4b.saveExtraction(outfile='Extract_4B.pkl')

    
if (doarc):
    arcfile = '/Users/simcoe/GIT/LlamasPipeline/Test/FinalAlignmentData/Bench_1a_Final_Alignment_Data/Green/REF_fIFU_Arc/Kr/20240506_134721_Camera3_1A_Green_0.5s-2.0e_Kr_Short.fits'
    arcsol = ArcLlamas(arcfile,trace1)

