//Convert ROIs generated in ABBA (Aligning Big Brains & Atlases) to tif masks
v = newArray("PTLp","ACA","ILA","AI","FRP","AUD","RSP","LSX","sAMY","MB","SSp-bfd",
"SSp-ll","SSp-m","SSp-n","MY","SSp-tr","SSp-ul","SSs","STRd","STRv","CBN","CBX",
"TEa","CLA","VIS","VISC","OLF","CTXsp","ORB","P","cc","cst","PAL","RHP","DORpm",
"DORsm","ECT","lfbst","PERI","cm","PL","MOp","mfbs","MOs","eps","GU","HIP","HY","SSp-un", "Right", "Left");

list = getFileList("path/to/slices/");

for (z=0; z<list.length; z++) {
	if (roiManager("Count")>0){
		roiManager("Select", Array.getSequence(roiManager("Count")));
		roiManager("Delete");
	}
	open("path/to/slices/"+list[z]);
	roiManager("Open", "path/to/rois/"+list[z]+".zip");
	for (i=0; i<v.length; i++) {
		roiName=v[i];
		nR = roiManager("Count"); 
		for (k=0; k<nR; k++) {
			roiManager("Select", k); 
			rName = Roi.getName(); 
			if (matches(rName, roiName)) { 
				all_rois = Array.getSequence(nR);
				without = Array.deleteIndex(all_rois,k);
				roiManager("Select", without);
				roiManager("Delete");
				run("Binary (0-255) mask(s) from Roi(s)", "save_mask(s) save_in=path/to/masks/ suffix=["+roiName+"] save_mask_as=tif]");
				roiManager("Select", 0);
				roiManager("Delete");
				roiManager("Open", "path/to/rois/"+list[z]+".zip");
			} 
		} 
	}
}