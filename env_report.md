# Ortam Raporu (env\_report.md)

## Sistem Bilgisi

* Ýþletim Sistemi: Windows 11, 23H2
* Kullanýcý: 90536

## Python ve Sanal Ortam

* Python sürümü: 3.10.0
* Sanal ortam: 
  venv klasörü altýnda kuruldu ve aktif edildi
* pip sürümü: 25.1.1

## Paketler

* torch (PyTorch) sürümü: 2.7.1 (CPU only)
* numpy sürümü: 2.2.6

## CUDA / GPU Durumu

* CUDA kullanýlabilir deðil (	torch.cuda.is\_available() = False)
* GPU: AMD Radeon(TM) Graphics, CUDA desteklemiyor.

## Sorunlar ve Çözümler

* PowerShell’de script çalýþtýrma engelini Set-ExecutionPolicy komutuyla aþtým.
* CUDA desteklenmediði için CPU sürümü PyTorch kullanýldý.
