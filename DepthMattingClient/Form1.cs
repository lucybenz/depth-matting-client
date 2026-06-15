using System.Diagnostics;
using System.Runtime.InteropServices;
using K4AdotNet;
using K4AdotNet.Sensor;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using OpenCvSharp;
using OpenCvSharp.Extensions;
using CvSize = OpenCvSharp.Size;
using K4Image = K4AdotNet.Sensor.Image;
using WinImage = System.Drawing.Image;

namespace DepthMattingClient;

public partial class Form1 : Form
{
    private readonly string _rootDir;
    private readonly string _modelDir;
    private readonly string _captureDir;
    private readonly PictureBox _rgbPreview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(20, 20, 20), SizeMode = PictureBoxSizeMode.Zoom };
    private readonly PictureBox _depthPreview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(20, 20, 20), SizeMode = PictureBoxSizeMode.Zoom };
    private readonly PictureBox _mattePreview = new() { Dock = DockStyle.Fill, BackColor = Color.FromArgb(20, 20, 20), SizeMode = PictureBoxSizeMode.Zoom };
    private readonly ComboBox _deviceList = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 280, Height = 52 };
    private readonly ComboBox _backgroundMode = new() { DropDownStyle = ComboBoxStyle.DropDownList, Width = 180, Height = 52 };
    private readonly Button _refreshButton = new() { Text = "Refresh Devices", Width = 185, Height = 52 };
    private readonly Button _startButton = new() { Text = "Start Depth", Width = 160, Height = 52, Enabled = false };
    private readonly Button _stopButton = new() { Text = "Stop", Width = 115, Height = 52, Enabled = false };
    private readonly Button _saveButton = new() { Text = "Save Current 4K PNG", Width = 210, Height = 52, Enabled = true };
    private readonly Button _openOutputButton = new() { Text = "Open Output Folder", Width = 190, Height = 52, Enabled = true };
    private readonly Button _pickBgImageButton = new() { Text = "BG Image", Width = 130, Height = 52 };
    private readonly Button _pickBgVideoButton = new() { Text = "BG Video", Width = 130, Height = 52 };
    private readonly NumericUpDown _previewMaxInput = new() { Minimum = 360, Maximum = 2160, Value = 960, Increment = 120, Width = 120, Height = 52 };
    private readonly NumericUpDown _downsampleInput = new() { Minimum = 0.05M, Maximum = 1.0M, DecimalPlaces = 3, Increment = 0.025M, Value = 0.25M, Width = 120, Height = 52 };
    private readonly NumericUpDown _depthCenterInput = new() { Minimum = 500, Maximum = 8000, Value = 3000, Increment = 100, Width = 130, Height = 52 };
    private readonly NumericUpDown _depthThicknessInput = new() { Minimum = 300, Maximum = 8000, Value = 3800, Increment = 100, Width = 130, Height = 52 };
    private readonly NumericUpDown _depthHoleFillInput = new() { Minimum = 0, Maximum = 80, Value = 35, Increment = 5, Width = 120, Height = 52 };
    private readonly CheckBox _useRvmBox = new() { Text = "RVM AI Matting: edge/hair", Checked = true, AutoSize = false, Width = 260, Height = 52, TextAlign = ContentAlignment.MiddleLeft };
    private readonly CheckBox _useDepthBox = new() { Text = "Depth Distance Filter: body range", Checked = true, AutoSize = false, Width = 300, Height = 52, TextAlign = ContentAlignment.MiddleLeft };
    private readonly CheckBox _showMaskBox = new() { Text = "Show Depth Mask Only", Checked = false, AutoSize = false, Width = 230, Height = 52, TextAlign = ContentAlignment.MiddleLeft };
    private readonly CheckBox _useGpuBox = new() { Text = "DirectML GPU", Checked = true, AutoSize = false, Width = 165, Height = 52, TextAlign = ContentAlignment.MiddleLeft };
    private readonly Label _status = new() { AutoSize = false, Dock = DockStyle.Bottom, Height = 42, TextAlign = ContentAlignment.MiddleLeft };
    private readonly ToolTip _tips = new() { AutoPopDelay = 12000, InitialDelay = 300, ReshowDelay = 100 };

    private readonly object _latestLock = new();
    private readonly object _backgroundLock = new();
    private readonly object _modelLock = new();
    private Mat? _latestColor;
    private Mat? _latestDepth16;
    private Mat? _latestOutput;
    private Mat? _backgroundImage;
    private VideoCapture? _backgroundVideo;
    private readonly Color _backgroundColor = Color.LimeGreen;
    private CancellationTokenSource? _cts;
    private Task? _captureTask;
    private RvmOnnxMatting? _matting;
    private volatile bool _running;
    private int _detectedDeviceCount;
    private int _frames;
    private int _previewUpdateBusy;
    private readonly Stopwatch _fpsClock = Stopwatch.StartNew();
    private readonly Stopwatch _previewClock = Stopwatch.StartNew();

    public Form1()
    {
        InitializeComponent();
        _rootDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", ".."));
        _modelDir = Path.Combine(_rootDir, "models");
        _captureDir = Path.Combine(_rootDir, "outputs", "captures");
        Directory.CreateDirectory(_modelDir);
        Directory.CreateDirectory(_captureDir);

        Text = "Depth RVM Matting Client - Femto Bolt";
        Width = 1500;
        Height = 920;
        MinimumSize = new System.Drawing.Size(1180, 760);

        BuildUi();
        WireEvents();
        SetStatus("Ready. Click Refresh Devices, then Start Depth.");
    }

    protected override void OnFormClosing(FormClosingEventArgs e)
    {
        StopPipeline();
        lock (_backgroundLock)
        {
            _backgroundImage?.Dispose();
            _backgroundVideo?.Dispose();
            _backgroundImage = null;
            _backgroundVideo = null;
        }

        lock (_modelLock)
        {
            _matting?.Dispose();
            _matting = null;
        }

        base.OnFormClosing(e);
    }

    protected override void OnShown(EventArgs e)
    {
        base.OnShown(e);
        SetStatus("Window ready. Click Refresh Devices. Model loads when Start Depth is clicked.");
    }

    private void BuildUi()
    {
        Font = new Font("Segoe UI", 11F, FontStyle.Regular, GraphicsUnit.Point);
        _backgroundMode.Items.AddRange(["Solid Color", "Image", "Video"]);
        _backgroundMode.SelectedIndex = 0;

        var top = new FlowLayoutPanel
        {
            Dock = DockStyle.Top,
            Height = 285,
            FlowDirection = FlowDirection.LeftToRight,
            WrapContents = true,
            Padding = new Padding(12),
            BackColor = Color.FromArgb(245, 245, 245)
        };

        top.Controls.AddRange([
            LabelOf("Device"), _deviceList, _refreshButton,
            LabelOf("Preview Max"), _previewMaxInput,
            LabelOf("RVM S"), _downsampleInput,
            LabelOf("Center Distance mm"), _depthCenterInput,
            LabelOf("Depth Thickness mm"), _depthThicknessInput,
            LabelOf("Depth Hole Fill"), _depthHoleFillInput,
            _useRvmBox, _useDepthBox, _showMaskBox, _useGpuBox,
            _startButton, _stopButton, _saveButton, _openOutputButton,
            LabelOf("Background"), _backgroundMode, _pickBgImageButton, _pickBgVideoButton
        ]);

        _tips.SetToolTip(_useRvmBox, "RVM AI matting controls fine edges, hair, clothes, and alpha detail.");
        _tips.SetToolTip(_useDepthBox, "Depth distance filter keeps pixels within Center Distance +/- half Thickness.");
        _tips.SetToolTip(_showMaskBox, "Shows the depth mask only. White is kept; black is removed.");
        _tips.SetToolTip(_depthCenterInput, "Expected person distance from the camera, in millimeters.");
        _tips.SetToolTip(_depthThicknessInput, "How thick the accepted depth range is. Increase this if body parts disappear.");
        _tips.SetToolTip(_depthHoleFillInput, "Fills holes in the depth mask. Increase for black pants or missing leg segments.");
        _tips.SetToolTip(_saveButton, "Saves current full-resolution RGB-D matting result as PNG files.");

        var grid = new TableLayoutPanel { Dock = DockStyle.Fill, ColumnCount = 3, RowCount = 2 };
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33F));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.33F));
        grid.ColumnStyles.Add(new ColumnStyle(SizeType.Percent, 33.34F));
        grid.RowStyles.Add(new RowStyle(SizeType.Absolute, 32));
        grid.RowStyles.Add(new RowStyle(SizeType.Percent, 100));
        grid.Controls.Add(Header("RGB"), 0, 0);
        grid.Controls.Add(Header("Depth"), 1, 0);
        grid.Controls.Add(Header("Matting"), 2, 0);
        grid.Controls.Add(_rgbPreview, 0, 1);
        grid.Controls.Add(_depthPreview, 1, 1);
        grid.Controls.Add(_mattePreview, 2, 1);

        Controls.Add(grid);
        Controls.Add(_status);
        Controls.Add(top);
    }

    private static Label LabelOf(string text) => new()
    {
        Text = text,
        AutoSize = false,
        Width = Math.Max(120, text.Length * 11),
        Height = 52,
        TextAlign = ContentAlignment.MiddleCenter,
        Margin = new Padding(12, 4, 2, 12)
    };

    private static Label Header(string text) => new()
    {
        Text = text,
        Dock = DockStyle.Fill,
        TextAlign = ContentAlignment.MiddleCenter,
        BackColor = Color.FromArgb(232, 232, 232)
    };

    private void WireEvents()
    {
        _refreshButton.Click += async (_, _) => await RefreshDevicesAsync();
        _startButton.Click += async (_, _) => await StartPipelineAsync();
        _stopButton.Click += (_, _) => StopPipeline();
        _saveButton.Click += async (_, _) => await SaveCurrentFrameAsync();
        _openOutputButton.Click += (_, _) => OpenOutputDirectory(_captureDir);
        _pickBgImageButton.Click += (_, _) => PickBackgroundImage();
        _pickBgVideoButton.Click += (_, _) => PickBackgroundVideo();
        _useGpuBox.CheckedChanged += async (_, _) => await TryLoadModelAsync(forceReload: true);
        _useDepthBox.CheckedChanged += (_, _) => SetStatus(_useDepthBox.Checked
            ? "Depth Distance Filter enabled: depth range will help remove background."
            : "Depth Distance Filter disabled: output uses RVM/result without depth range filtering.");
        _showMaskBox.CheckedChanged += (_, _) => SetStatus(_showMaskBox.Checked
            ? "Depth mask preview enabled: white area is kept, black area is removed."
            : "Depth mask preview disabled: showing final matting result.");
        _useRvmBox.CheckedChanged += async (_, _) =>
        {
            if (_useRvmBox.Checked)
            {
                await TryLoadModelAsync();
            }
            else
            {
                SetStatus("RVM disabled. Depth-only/background comparison mode.");
            }
        };
    }

    private async Task RefreshDevicesAsync()
    {
        _refreshButton.Enabled = false;
        SetStatus("Refreshing Femto devices...");
        try
        {
            _deviceList.Items.Clear();
            var count = 0;
            var queryTask = Task.Run(() => Device.InstalledCount);
            var finished = await Task.WhenAny(queryTask, Task.Delay(TimeSpan.FromSeconds(5)));
            if (finished != queryTask)
            {
                SetStatus("Device query timed out. Check Femto driver/USB connection, then try again.");
                return;
            }

            count = queryTask.Result;

            _detectedDeviceCount = count;
            for (var i = 0; i < Math.Max(count, 1); i++)
            {
                _deviceList.Items.Add(new DepthDeviceInfo(i, i < count ? $"Femto Device {i}" : "Device 0"));
            }

            _deviceList.SelectedIndex = 0;
            _startButton.Enabled = count > 0;
            SetStatus(count > 0 ? $"Detected {count} Femto/K4A device(s)." : "No Femto device detected yet. Connect the camera, then Refresh.");
        }
        catch (Exception ex)
        {
            AppLog.Write(ex);
            SetStatus($"Device query failed: {ex.Message}");
        }
        finally
        {
            _refreshButton.Enabled = true;
        }
    }

    private async Task TryLoadModelAsync(bool forceReload = false)
    {
        if (_running && forceReload)
        {
            SetStatus("Stop streaming before switching GPU provider.");
            return;
        }

        lock (_modelLock)
        {
            if (!forceReload && _matting != null)
            {
                return;
            }
        }

        var useGpu = _useGpuBox.Checked;
        SetStatus(useGpu ? "Loading RVM model on DirectML..." : "Loading RVM model on CPU...");
        var modelPath = Directory.GetFiles(_modelDir, "*.onnx").FirstOrDefault()
            ?? Path.Combine(Path.GetDirectoryName(_rootDir) ?? _rootDir, "native_matting_client", "models", "rvm_mobilenetv3_fp32.onnx");

        if (!File.Exists(modelPath))
        {
            SetStatus($"RVM ONNX not found. Put model in {Path.GetFullPath(_modelDir)}.");
            return;
        }

        RvmOnnxMatting? next = null;
        RvmOnnxMatting? old = null;
        try
        {
            next = await Task.Run(() => new RvmOnnxMatting(modelPath, useGpu));
            lock (_modelLock)
            {
                old = _matting;
                _matting = next;
                next = null;
            }

            old?.Dispose();
            SetStatus($"RVM loaded: {Path.GetFileName(modelPath)} | {_matting?.ProviderName}");
        }
        catch (Exception ex)
        {
            next?.Dispose();
            AppLog.Write(ex);
            SetStatus($"RVM load failed: {ex.Message}");
        }
    }

    private RvmOnnxMatting? CurrentModel()
    {
        lock (_modelLock)
        {
            return _matting;
        }
    }

    private string? CurrentModelPath()
    {
        lock (_modelLock)
        {
            return _matting?.ModelPath;
        }
    }

    private async Task StartPipelineAsync()
    {
        if (_running || _deviceList.SelectedItem is not DepthDeviceInfo device)
        {
            return;
        }

        if (_detectedDeviceCount <= 0)
        {
            SetStatus("No Femto device detected. Click Refresh Devices after connecting the camera.");
            return;
        }

        _startButton.Enabled = false;
        if (_useRvmBox.Checked)
        {
            await TryLoadModelAsync();
        }
        _running = true;
        _cts = new CancellationTokenSource();
        _captureTask = Task.Run(() => CaptureLoop(device.Index, _cts.Token));
        _stopButton.Enabled = true;
    }

    private void StopPipeline()
    {
        if (!_running)
        {
            return;
        }

        _running = false;
        _cts?.Cancel();
        try
        {
            _captureTask?.Wait(TimeSpan.FromSeconds(2));
        }
        catch
        {
            // Device stop can race with a blocking capture call.
        }

        _cts?.Dispose();
        _cts = null;
        _captureTask = null;

        lock (_latestLock)
        {
            _latestColor?.Dispose();
            _latestDepth16?.Dispose();
            _latestOutput?.Dispose();
            _latestColor = null;
            _latestDepth16 = null;
            _latestOutput = null;
        }

        _startButton.Enabled = true;
        _stopButton.Enabled = false;
        SetStatus("Stopped.");
    }

    private void CaptureLoop(int index, CancellationToken token)
    {
        Device? device = null;
        try
        {
            device = Device.Open(index);
            BeginInvoke(() => SetStatus($"Opened {device}. Starting cameras..."));
            var config = DeviceConfiguration.DisableAll;
            config.ColorFormat = ImageFormat.ColorBgra32;
            config.ColorResolution = ColorResolution.R2160p;
            config.DepthMode = DepthMode.NarrowViewUnbinned;
            config.CameraFps = FrameRate.Thirty;
            config.SynchronizedImagesOnly = true;
            if (!config.IsValid(out var reason))
            {
                BeginInvoke(() => SetStatus($"Invalid camera config: {reason}"));
                _running = false;
                return;
            }

            device.SetSoftFilter(true);
            device.StartCameras(config);
            BeginInvoke(() => SetStatus($"Streaming {device.SerialNumber} | 4K RGB + Depth"));

            while (!token.IsCancellationRequested)
            {
                if (!device.TryGetCapture(out var capture, new K4AdotNet.Timeout(50)) || capture == null)
                {
                    continue;
                }

                using (capture)
                using (var colorImage = capture.ColorImage)
                using (var depthImage = capture.DepthImage)
                {
                    if (colorImage == null || depthImage == null)
                    {
                        continue;
                    }

                    using var color = K4ColorToBgr(colorImage);
                    using var depth16 = K4DepthToMat(depthImage);
                    using var previewColor = ResizeLongest(color, (int)_previewMaxInput.Value);
                    using var previewDepth = ResizeLongest(depth16, (int)_previewMaxInput.Value);
                    var output = ProcessMatting(previewColor, previewDepth);

                    lock (_latestLock)
                    {
                        _latestColor?.Dispose();
                        _latestDepth16?.Dispose();
                        _latestOutput?.Dispose();
                        _latestColor = color.Clone();
                        _latestDepth16 = depth16.Clone();
                        _latestOutput = output.Clone();
                    }

                    if (ShouldUpdatePreview())
                    {
                        using var depthVisual = VisualizeDepth(previewDepth);
                        PostPreviews(previewColor, depthVisual, output);
                    }
                    output.Dispose();
                    Interlocked.Increment(ref _frames);
                    PostStats(color.Width, color.Height, depth16.Width, depth16.Height);
                }
            }
        }
        catch (Exception ex)
        {
            AppLog.Write(ex);
            BeginInvoke(() => SetStatus($"Depth pipeline error: {ex.Message}"));
        }
        finally
        {
            try
            {
                device?.StopCameras();
            }
            catch
            {
                // Camera may already be stopped.
            }

            device?.Dispose();
            BeginInvoke(() =>
            {
                _running = false;
                _startButton.Enabled = true;
                _stopButton.Enabled = false;
            });
        }
    }

    private Mat ProcessMatting(Mat colorBgr, Mat depth16)
    {
        using var bg = PrepareBackground(colorBgr.Width, colorBgr.Height);
        using var depthMask = DepthForegroundMask(depth16, colorBgr.Width, colorBgr.Height);
        if (_showMaskBox.Checked)
        {
            using var maskPreview = new Mat();
            Cv2.CvtColor(depthMask, maskPreview, ColorConversionCodes.GRAY2BGR);
            return maskPreview.Clone();
        }

        var matting = CurrentModel();
        Mat result;
        if (_useRvmBox.Checked && matting != null)
        {
            result = matting.Matte(colorBgr, (float)_downsampleInput.Value, bg);
            if (!_useDepthBox.Checked)
            {
                return result;
            }
        }
        else if (_useDepthBox.Checked)
        {
            result = colorBgr.Clone();
        }
        else
        {
            return colorBgr.Clone();
        }

        using var softened = RefineDepthMask(depthMask, (int)_depthHoleFillInput.Value);
        using var resultF = new Mat();
        using var bgF = new Mat();
        result.ConvertTo(resultF, MatType.CV_32FC3, 1.0 / 255.0);
        bg.ConvertTo(bgF, MatType.CV_32FC3, 1.0 / 255.0);
        using var maskF1 = new Mat();
        softened.ConvertTo(maskF1, MatType.CV_32FC1, 1.0 / 255.0);
        using var maskF3 = new Mat();
        Cv2.CvtColor(maskF1, maskF3, ColorConversionCodes.GRAY2BGR);
        using var inv = new Mat(maskF3.Size(), maskF3.Type(), Scalar.All(1));
        Cv2.Subtract(inv, maskF3, inv);
        using var a = new Mat();
        using var b = new Mat();
        Cv2.Multiply(resultF, maskF3, a);
        Cv2.Multiply(bgF, inv, b);
        using var merged = new Mat();
        Cv2.Add(a, b, merged);
        result.Dispose();
        var output = new Mat();
        merged.ConvertTo(output, MatType.CV_8UC3, 255.0);
        return output;
    }

    private Mat DepthForegroundMask(Mat depth16, int width, int height)
    {
        using var resized = new Mat();
        Cv2.Resize(depth16, resized, new CvSize(width, height), 0, 0, InterpolationFlags.Nearest);
        var mask = new Mat(height, width, MatType.CV_8UC1, Scalar.Black);
        var center = (int)_depthCenterInput.Value;
        var thickness = (int)_depthThicknessInput.Value;
        var half = Math.Max(1, thickness / 2);
        var minDepth = Math.Max(1, center - half);
        var maxDepth = Math.Min(9000, center + half);
        Cv2.InRange(resized, new Scalar(minDepth), new Scalar(maxDepth), mask);
        return mask;
    }

    private Mat DepthForegroundMask(Mat depth16, int width, int height, int center, int thickness)
    {
        using var resized = new Mat();
        Cv2.Resize(depth16, resized, new CvSize(width, height), 0, 0, InterpolationFlags.Nearest);
        var mask = new Mat(height, width, MatType.CV_8UC1, Scalar.Black);
        var half = Math.Max(1, thickness / 2);
        var minDepth = Math.Max(1, center - half);
        var maxDepth = Math.Min(9000, center + half);
        Cv2.InRange(resized, new Scalar(minDepth), new Scalar(maxDepth), mask);
        return mask;
    }

    private Mat ProcessMattingForSave(
        Mat colorBgr,
        Mat depth16,
        bool useRvm,
        bool useDepth,
        bool showMask,
        float downsample,
        string? modelPath,
        int center,
        int thickness,
        int holeFill,
        int bgMode,
        bool useGpu)
    {
        using var bg = PrepareBackgroundForSave(colorBgr.Width, colorBgr.Height, bgMode);
        using var depthMask = DepthForegroundMask(depth16, colorBgr.Width, colorBgr.Height, center, thickness);
        if (showMask)
        {
            using var maskPreview = new Mat();
            Cv2.CvtColor(depthMask, maskPreview, ColorConversionCodes.GRAY2BGR);
            return maskPreview.Clone();
        }

        Mat result;
        if (useRvm && !string.IsNullOrWhiteSpace(modelPath) && File.Exists(modelPath))
        {
            using var saveModel = new RvmOnnxMatting(modelPath, useGpu);
            result = saveModel.Matte(colorBgr, downsample, bg);
            if (!useDepth)
            {
                return result;
            }
        }
        else if (useDepth)
        {
            result = colorBgr.Clone();
        }
        else
        {
            return colorBgr.Clone();
        }

        using var softened = RefineDepthMask(depthMask, holeFill);
        using var resultF = new Mat();
        using var bgF = new Mat();
        result.ConvertTo(resultF, MatType.CV_32FC3, 1.0 / 255.0);
        bg.ConvertTo(bgF, MatType.CV_32FC3, 1.0 / 255.0);
        using var maskF1 = new Mat();
        softened.ConvertTo(maskF1, MatType.CV_32FC1, 1.0 / 255.0);
        using var maskF3 = new Mat();
        Cv2.CvtColor(maskF1, maskF3, ColorConversionCodes.GRAY2BGR);
        using var inv = new Mat(maskF3.Size(), maskF3.Type(), Scalar.All(1));
        Cv2.Subtract(inv, maskF3, inv);
        using var a = new Mat();
        using var b = new Mat();
        Cv2.Multiply(resultF, maskF3, a);
        Cv2.Multiply(bgF, inv, b);
        using var merged = new Mat();
        Cv2.Add(a, b, merged);
        result.Dispose();
        var output = new Mat();
        merged.ConvertTo(output, MatType.CV_8UC3, 255.0);
        return output;
    }

    private Mat PrepareBackgroundForSave(int width, int height, int bgMode)
    {
        var bg = new Mat();
        lock (_backgroundLock)
        {
            if (bgMode == 1 && _backgroundImage != null)
            {
                Cv2.Resize(_backgroundImage, bg, new CvSize(width, height), 0, 0, InterpolationFlags.Linear);
                return bg;
            }

            if (bgMode == 2 && _backgroundVideo != null)
            {
                using var frame = new Mat();
                if (!_backgroundVideo.Read(frame) || frame.Empty())
                {
                    _backgroundVideo.Set(VideoCaptureProperties.PosFrames, 0);
                    _backgroundVideo.Read(frame);
                }

                if (!frame.Empty())
                {
                    Cv2.Resize(frame, bg, new CvSize(width, height), 0, 0, InterpolationFlags.Linear);
                    return bg;
                }
            }
        }

        bg.Create(height, width, MatType.CV_8UC3);
        bg.SetTo(new Scalar(_backgroundColor.B, _backgroundColor.G, _backgroundColor.R));
        return bg;
    }

    private static Mat RefineDepthMask(Mat mask, int holeFill)
    {
        var refined = new Mat();
        var closeSize = MakeOdd(Math.Clamp(holeFill, 5, 101));
        var dilateSize = MakeOdd(Math.Clamp(holeFill / 2, 3, 51));
        var blurSize = MakeOdd(Math.Clamp(holeFill, 9, 101));
        using var closeKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new CvSize(closeSize, closeSize));
        using var dilateKernel = Cv2.GetStructuringElement(MorphShapes.Ellipse, new CvSize(dilateSize, dilateSize));
        Cv2.MorphologyEx(mask, refined, MorphTypes.Close, closeKernel);
        Cv2.Dilate(refined, refined, dilateKernel, iterations: 1);
        Cv2.GaussianBlur(refined, refined, new CvSize(blurSize, blurSize), 0);
        return refined;
    }

    private static int MakeOdd(int value) => value % 2 == 0 ? value + 1 : value;

    private Mat PrepareBackground(int width, int height)
    {
        var bg = new Mat();
        lock (_backgroundLock)
        {
            if (_backgroundMode.SelectedIndex == 1 && _backgroundImage != null)
            {
                Cv2.Resize(_backgroundImage, bg, new CvSize(width, height), 0, 0, InterpolationFlags.Linear);
                return bg;
            }

            if (_backgroundMode.SelectedIndex == 2 && _backgroundVideo != null)
            {
                using var frame = new Mat();
                if (!_backgroundVideo.Read(frame) || frame.Empty())
                {
                    _backgroundVideo.Set(VideoCaptureProperties.PosFrames, 0);
                    _backgroundVideo.Read(frame);
                }

                if (!frame.Empty())
                {
                    Cv2.Resize(frame, bg, new CvSize(width, height), 0, 0, InterpolationFlags.Linear);
                    return bg;
                }
            }
        }

        bg.Create(height, width, MatType.CV_8UC3);
        bg.SetTo(new Scalar(_backgroundColor.B, _backgroundColor.G, _backgroundColor.R));
        return bg;
    }

    private void PickBackgroundImage()
    {
        using var dialog = new OpenFileDialog { Filter = "Image|*.jpg;*.jpeg;*.png;*.bmp|All files|*.*" };
        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        var image = Cv2.ImRead(dialog.FileName, ImreadModes.Color);
        if (image.Empty())
        {
            SetStatus("Background image load failed.");
            return;
        }

        lock (_backgroundLock)
        {
            _backgroundImage?.Dispose();
            _backgroundImage = image;
        }

        _backgroundMode.SelectedIndex = 1;
        SetStatus($"Background image loaded: {dialog.FileName}");
    }

    private void PickBackgroundVideo()
    {
        using var dialog = new OpenFileDialog { Filter = "Video|*.mp4;*.mov;*.avi;*.mkv|All files|*.*" };
        if (dialog.ShowDialog(this) != DialogResult.OK)
        {
            return;
        }

        var video = new VideoCapture(dialog.FileName);
        if (!video.IsOpened())
        {
            video.Dispose();
            SetStatus("Background video open failed.");
            return;
        }

        lock (_backgroundLock)
        {
            _backgroundVideo?.Dispose();
            _backgroundVideo = video;
        }

        _backgroundMode.SelectedIndex = 2;
        SetStatus($"Background video loaded: {dialog.FileName}");
    }

    private async Task SaveCurrentFrameAsync()
    {
        _saveButton.Enabled = false;
        SetStatus("Preparing current RGB-D frame for 4K PNG save...");
        try
        {
            Mat? color;
            Mat? depth;
            Mat? previewOutput;
            lock (_latestLock)
            {
                color = _latestColor?.Clone();
                depth = _latestDepth16?.Clone();
                previewOutput = _latestOutput?.Clone();
            }

            if (color == null || depth == null)
            {
                SetStatus("No RGB-D frame to save yet. Start Depth first and wait until preview appears.");
                return;
            }

            var useRvm = _useRvmBox.Checked;
            var useDepth = _useDepthBox.Checked;
            var showMask = _showMaskBox.Checked;
            var downsample = (float)_downsampleInput.Value;
            var modelPath = CurrentModelPath();
            var center = (int)_depthCenterInput.Value;
            var thickness = (int)_depthThicknessInput.Value;
            var holeFill = (int)_depthHoleFillInput.Value;
            var bgMode = _backgroundMode.SelectedIndex;
            var useGpu = _useGpuBox.Checked;

            var result = await Task.Run(() =>
            {
                using (color)
                using (depth)
                {
                    Mat output;
                    try
                    {
                        output = ProcessMattingForSave(color, depth, useRvm, useDepth, showMask, downsample, modelPath, center, thickness, holeFill, bgMode, useGpu);
                    }
                    catch (Exception ex)
                    {
                        AppLog.Write(ex);
                        if (previewOutput == null || previewOutput.Empty())
                        {
                            throw;
                        }

                        output = new Mat();
                        Cv2.Resize(previewOutput, output, new CvSize(color.Width, color.Height), 0, 0, InterpolationFlags.Lanczos4);
                    }

                    using (output)
                    using (var depthVisual = VisualizeDepth(depth))
                    {
                        var stamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                        var mattePath = Path.Combine(_captureDir, $"matting_result_4k_{stamp}.png");
                        var rgbPath = Path.Combine(_captureDir, $"source_rgb_4k_{stamp}.png");
                        var depthPath = Path.Combine(_captureDir, $"depth_debug_{stamp}.png");
                        Cv2.ImWrite(mattePath, output);
                        Cv2.ImWrite(rgbPath, color);
                        Cv2.ImWrite(depthPath, depthVisual);
                        return mattePath;
                    }
                }
            });

            previewOutput?.Dispose();

            AppLog.WriteText($"Saved matting result: {result}");
            SetStatus($"Saved matting result: {result}");
            OpenOutputDirectory(result);
        }
        catch (Exception ex)
        {
            AppLog.Write(ex);
            SetStatus($"Save failed: {ex.Message}");
        }
        finally
        {
            _saveButton.Enabled = true;
        }
    }

    private static void OpenOutputDirectory(string path)
    {
        try
        {
            var arguments = Directory.Exists(path) ? $"\"{path}\"" : $"/select,\"{path}\"";
            Process.Start(new ProcessStartInfo
            {
                FileName = "explorer.exe",
                Arguments = arguments,
                UseShellExecute = true
            });
        }
        catch
        {
            // Opening Explorer is convenience only.
        }
    }

    private static Mat K4ColorToBgr(K4Image image)
    {
        var bytes = new byte[image.SizeBytes];
        image.CopyTo(bytes);
        using var bgra = new Mat(image.HeightPixels, image.WidthPixels, MatType.CV_8UC4);
        Marshal.Copy(bytes, 0, bgra.Data, bytes.Length);
        var bgr = new Mat();
        Cv2.CvtColor(bgra, bgr, ColorConversionCodes.BGRA2BGR);
        return bgr;
    }

    private static Mat K4DepthToMat(K4Image image)
    {
        var values = new short[image.SizeBytes / 2];
        image.CopyTo(values);
        var depth = new Mat(image.HeightPixels, image.WidthPixels, MatType.CV_16UC1);
        Marshal.Copy(values, 0, depth.Data, values.Length);
        return depth;
    }

    private static Mat VisualizeDepth(Mat depth16)
    {
        using var depth8 = new Mat();
        depth16.ConvertTo(depth8, MatType.CV_8UC1, 255.0 / 6000.0);
        var colored = new Mat();
        Cv2.ApplyColorMap(depth8, colored, ColormapTypes.Turbo);
        return colored;
    }

    private static Mat ResizeLongest(Mat src, int maxSide)
    {
        var longest = Math.Max(src.Width, src.Height);
        if (longest <= maxSide)
        {
            return src.Clone();
        }

        var scale = maxSide / (double)longest;
        var dst = new Mat();
        Cv2.Resize(src, dst, new CvSize((int)Math.Round(src.Width * scale), (int)Math.Round(src.Height * scale)), 0, 0,
            src.Type() == MatType.CV_16UC1 ? InterpolationFlags.Nearest : InterpolationFlags.Area);
        return dst;
    }

    private bool ShouldUpdatePreview()
    {
        if (_previewClock.ElapsedMilliseconds < 100)
        {
            return false;
        }

        if (Interlocked.CompareExchange(ref _previewUpdateBusy, 1, 0) != 0)
        {
            return false;
        }

        _previewClock.Restart();
        return true;
    }

    private void PostPreviews(Mat rgb, Mat depth, Mat matte)
    {
        try
        {
            var rgbBitmap = BitmapConverter.ToBitmap(rgb);
            var depthBitmap = BitmapConverter.ToBitmap(depth);
            var matteBitmap = BitmapConverter.ToBitmap(matte);
            BeginInvoke(() =>
            {
                SwapImage(_rgbPreview, rgbBitmap);
                SwapImage(_depthPreview, depthBitmap);
                SwapImage(_mattePreview, matteBitmap);
                Interlocked.Exchange(ref _previewUpdateBusy, 0);
            });
        }
        catch
        {
            Interlocked.Exchange(ref _previewUpdateBusy, 0);
            // Preview errors should not stop the device loop.
        }
    }

    private static void SwapImage(PictureBox target, WinImage next)
    {
        var old = target.Image;
        target.Image = next;
        old?.Dispose();
    }

    private void PostStats(int colorW, int colorH, int depthW, int depthH)
    {
        if (_fpsClock.Elapsed.TotalSeconds < 1)
        {
            return;
        }

        var fps = Interlocked.Exchange(ref _frames, 0) / _fpsClock.Elapsed.TotalSeconds;
        _fpsClock.Restart();
        var center = (int)_depthCenterInput.Value;
        var thickness = (int)_depthThicknessInput.Value;
        var holeFill = (int)_depthHoleFillInput.Value;
        var minDepth = Math.Max(1, center - Math.Max(1, thickness / 2));
        var maxDepth = Math.Min(9000, center + Math.Max(1, thickness / 2));
        var mode = $"RVM {(_useRvmBox.Checked ? "On" : "Off")} | Depth Assist {(_useDepthBox.Checked ? "On" : "Off")} | Mask Preview {(_showMaskBox.Checked ? "On" : "Off")}";
        BeginInvoke(() => SetStatus($"RGB {colorW}x{colorH} | Depth {depthW}x{depthH} | FPS {fps:F1} | Depth Range {minDepth}-{maxDepth}mm | Hole Fill {holeFill} | {mode} | Provider {(CurrentModel()?.ProviderName ?? "No model")}"));
    }

    private void SetStatus(string text)
    {
        _status.Text = "  " + text;
    }
}

public sealed record DepthDeviceInfo(int Index, string Name)
{
    public override string ToString() => $"{Name} [Index {Index}]";
}

public sealed class RvmOnnxMatting : IDisposable
{
    private readonly string _modelPath;
    private readonly bool _useGpu;
    private readonly InferenceSession _session;
    private DenseTensor<float>[] _rec = CreateInitialRec();
    private readonly string[] _inputNames;
    public string ProviderName { get; }
    public string ModelPath => _modelPath;

    public RvmOnnxMatting(string modelPath, bool useGpu)
    {
        _modelPath = modelPath;
        _useGpu = useGpu;
        var options = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            ExecutionMode = ExecutionMode.ORT_SEQUENTIAL,
            EnableMemoryPattern = false
        };

        if (useGpu)
        {
            options.AppendExecutionProvider_DML(0);
            ProviderName = "DirectML";
        }
        else
        {
            ProviderName = "CPU";
        }

        _session = new InferenceSession(modelPath, options);
        _inputNames = _session.InputMetadata.Keys.ToArray();
    }

    public Mat Matte(Mat bgr, float downsampleRatio, Mat bg)
    {
        var srcTensor = ToTensor(bgr);
        var downsample = new DenseTensor<float>(new[] { Math.Clamp(downsampleRatio, 0.05f, 1f) }, [1]);
        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor(InputName(0, "src"), srcTensor),
            NamedOnnxValue.CreateFromTensor(InputName(1, "r1i"), _rec[0]),
            NamedOnnxValue.CreateFromTensor(InputName(2, "r2i"), _rec[1]),
            NamedOnnxValue.CreateFromTensor(InputName(3, "r3i"), _rec[2]),
            NamedOnnxValue.CreateFromTensor(InputName(4, "r4i"), _rec[3]),
            NamedOnnxValue.CreateFromTensor(InputName(5, "downsample_ratio"), downsample)
        };

        using var results = _session.Run(inputs);
        var resultList = results.ToList();
        var fgr = resultList[0].AsTensor<float>();
        var pha = resultList[1].AsTensor<float>();
        _rec = resultList.Skip(2).Take(4).Select(v => CloneTensor(v.AsTensor<float>())).ToArray();
        return Composite(fgr, pha, bg);
    }

    private string InputName(int index, string fallback) => _inputNames.Length > index ? _inputNames[index] : fallback;

    private static DenseTensor<float>[] CreateInitialRec() =>
    [
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1]),
        new DenseTensor<float>(new[] { 0f }, [1, 1, 1, 1])
    ];

    private static DenseTensor<float> CloneTensor(Tensor<float> tensor)
    {
        var clone = new DenseTensor<float>(tensor.Dimensions.ToArray());
        tensor.ToArray().CopyTo(clone.Buffer.Span);
        return clone;
    }

    private static DenseTensor<float> ToTensor(Mat bgr)
    {
        var height = bgr.Height;
        var width = bgr.Width;
        var tensor = new DenseTensor<float>([1, 3, height, width]);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var p = bgr.At<Vec3b>(y, x);
                tensor[0, 0, y, x] = p.Item2 / 255f;
                tensor[0, 1, y, x] = p.Item1 / 255f;
                tensor[0, 2, y, x] = p.Item0 / 255f;
            }
        }

        return tensor;
    }

    private static Mat Composite(Tensor<float> fgr, Tensor<float> pha, Mat bg)
    {
        var height = bg.Height;
        var width = bg.Width;
        var output = new Mat(height, width, MatType.CV_8UC3);
        for (var y = 0; y < height; y++)
        {
            for (var x = 0; x < width; x++)
            {
                var alpha = Math.Clamp(pha[0, 0, y, x], 0f, 1f);
                var bgPixel = bg.At<Vec3b>(y, x);
                var fr = Math.Clamp(fgr[0, 0, y, x], 0f, 1f) * 255f;
                var fg = Math.Clamp(fgr[0, 1, y, x], 0f, 1f) * 255f;
                var fb = Math.Clamp(fgr[0, 2, y, x], 0f, 1f) * 255f;
                output.Set(y, x, new Vec3b(
                    (byte)Math.Clamp(fb * alpha + bgPixel.Item0 * (1f - alpha), 0f, 255f),
                    (byte)Math.Clamp(fg * alpha + bgPixel.Item1 * (1f - alpha), 0f, 255f),
                    (byte)Math.Clamp(fr * alpha + bgPixel.Item2 * (1f - alpha), 0f, 255f)));
            }
        }

        return output;
    }

    public void Dispose() => _session.Dispose();
}
