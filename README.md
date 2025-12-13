================================== Build vortex START ==================================

```
git clone --depth=1 --recursive https://github.com/jel221/vortex-spgemm.git
cd vortex-spgemm

mkdir build
cd build
../configure --xlen=32 --tooldir=$HOME/tools

source ./ci/toolchain_env.sh
make -s
```

================================== Build vortex END ==================================

You can either run the provided b.sh file

`bash ../b.sh`

Or manually run these commands:

```
make -C tests/regression/spgemm clean
CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32 -DDIM_N=64" make -C tests/regression/spgemm
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=64" ./ci/blackbox.sh --driver=simx --app=spgemm
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=64" ./ci/blackbox.sh --driver=rtlsim --app=spgemm

make -C tests/regression/spgemm clean
CONFIGS="-DNUM_THREADS=8 -DITYPE=fp16 -DOTYPE=fp32 -DDIM_N=32" make -C tests/regression/spgemm
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=32" ./ci/blackbox.sh --driver=simx --app=spgemm
CONFIGS="-DNUM_THREADS=8 -DEXT_TCU_ENABLE -DTCU_BHF -DDIM_N=32" ./ci/blackbox.sh --driver=rtlsim --app=spgemm
```