echo -e "file name\tChekSum\tUT\tOBJECT\tEXPTIME\tREAD-MDE\tAIRMASS" > obs.log
echo -en "---------------------------------------\t------\t------------\t" >> obs.log
echo -en "-----------------------------\t----------\t-----\n" >> obs.log
for i in `ls *mef.fits`; do   
    echo -en $i"\t"
#    echo -en `du -sh $i | sed s/M// | awk '{if ($1==198) {print "OK"} else {print "ERR"}}'`"\t"
    echo -en `du -sh $i | sed s/M// | awk '{if ($1==177) {print "OK"} else {print "ERR"}}'`"\t"
    echo -en `echo $i | sed s/.*T// | sed s/_mef.*//`
    echo -en "\t"
    echo -en `head -c 2880 $i | fold -w 80 | grep OBJECT | sed s/.*=.// | sed s/\'// | sed s/\'.*//`
    echo -en "\t"
    echo -en `head -c 2880 $i | fold -w 80 | grep REXPTIME | sed s/.*=// | sed s/...R.*//`
    echo -en "\t"
    echo -en `head -c 2880 $i | fold -w 80 | grep READ-MDE | sed s/.*=.// | sed s/\'// | sed s/\'.*// | xargs`
    echo -en "\t"
    echo -en `head -c 2880 $i | fold -w 80 | grep AIRMASS | sed s/.*=// | sed s/...O.*//`
    echo    
done >> obs.log
cat obs.log | column -t -s $'\t' 

