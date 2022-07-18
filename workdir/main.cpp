#include <iostream>
#include <vector>

#include "TFile.h"
#include "TTree.h"
#include "TString.h"

#include <pybind11/embed.h> // everything needed for embedding
#include <pybind11/eval.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py = pybind11;
using namespace pybind11::literals;


int main() {

  // prepare the interpreter and load modules
  
  py::scoped_interpreter guard{}; // start the interpreter and keep it alive

  py::module joblib = py::module::import("joblib");
  py::module_ np = py::module_::import("numpy");  // like 'import numpy as np'


  // load data
  
  TFile* f = new TFile("etaPi_2016_mu_new.root","READ");
  TTree* t = (TTree*)f->Get("DecayTree");
  t->Show(0);
  
  std::vector<TString> varnames;
  varnames.push_back("D_PVFitBeam_chi2"); 
  varnames.push_back("D_IPCHI2_OWNPV"); 
  varnames.push_back("D_ACosDira_OWNPV"); 
  varnames.push_back("eta_IPCHI2_OWNPV"); 
  varnames.push_back("maxPIDK");
  varnames.push_back("maxPIDe");
  varnames.push_back("maxPIDp");
  varnames.push_back("maxGhostProb");
  varnames.push_back("maxCHI2NDOF");
  varnames.push_back("gamma_PT");
  varnames.push_back("gamma_CL");
  varnames.push_back("rho_M");
  
  int len = varnames.size();
  Double_t* values = new Double_t[len];

  Float_t chi2 = 0;
  t->SetBranchAddress(varnames.at(0).Data(),&chi2);  
  for (int i=1; i<len; i++){
    t->SetBranchAddress(varnames.at(i).Data(),&values[i]);
  }


  // cycle over data 
  for (int j=0; j<20; j++){ // t->GetEntries(); j++){
    t->GetEntry(j);
    values[0] = (Double_t)chi2; 

    // convert the features to a numpy array with the right dimensions
    // <-- input features
    py::tuple tup = py::make_tuple(values[0],
				   values[1],
				   values[2],
				   values[3],
				   values[4],
				   values[5],
				   values[6],
				   values[7],
				   values[8],
				   values[9],
				   values[10],
				   values[11]
				   ); 

    py::array_t<float> arr = np.attr("array")(tup, "dtype"_a="float64");
    py::array_t<float> arr2d = arr.attr("reshape")(1, -1); 

    // load the classifier and predict the class for the input data
    py::object clf = joblib.attr("load")("etaPi_BDT_clf.joblib") ; 
    py::object res = clf.attr("predict")(arr2d);
    py::object prob = clf.attr("predict_proba")(arr2d); 
    int len = prob.attr("__len__")().cast<int>() ;
    int size = prob.attr("size").cast<int>() ;
    double prob0 = prob.attr("item")(0).cast<double>() ;
    double prob1 = prob.attr("item")(1).cast<double>() ;

      
    std::cout << " class is: " << res.cast<int>()
	      << " prob is: " << prob0 << " " << prob1 
	      << std::endl;
    
  }

  



}



