make -C tests/regression/spgemm clean
CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32 -DDIM_N=64" make -C tests/regression/spgemm

echo "Using N=64"
echo "SimX for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=64" ./ci/blackbox.sh --driver=simx --app=spgemm

echo "RTLSim for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=64" ./ci/blackbox.sh --driver=rtlsim --app=spgemm



echo "Using N=32"

make -C tests/regression/spgemm clean

CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32 -DDIM_N=32" make -C tests/regression/spgemm

echo "SimX for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=32" ./ci/blackbox.sh --driver=simx --app=spgemm

echo "RTLSim for sparse matmul"
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=32" ./ci/blackbox.sh --driver=rtlsim --app=spgemm


echo ""
echo "You can run other tests if you vary DIM_N in the Makefile CONFIGS"