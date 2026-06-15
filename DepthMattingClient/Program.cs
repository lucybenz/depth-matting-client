namespace DepthMattingClient;

static class Program
{
    /// <summary>
    ///  The main entry point for the application.
    /// </summary>
    [STAThread]
    static void Main()
    {
        Application.SetUnhandledExceptionMode(UnhandledExceptionMode.CatchException);
        Application.ThreadException += (_, e) => AppLog.Write(e.Exception);
        AppDomain.CurrentDomain.UnhandledException += (_, e) =>
        {
            if (e.ExceptionObject is Exception ex)
            {
                AppLog.Write(ex);
            }
            else
            {
                AppLog.WriteText("Unhandled non-Exception: " + e.ExceptionObject);
            }
        };

        try
        {
            ApplicationConfiguration.Initialize();
            Application.Run(new Form1());
        }
        catch (Exception ex)
        {
            AppLog.Write(ex);
            MessageBox.Show(ex.ToString(), "DepthMattingClient startup error", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }    
}

internal static class AppLog
{
    public static readonly string LogDir = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", "logs"));

    public static void Write(Exception ex) => WriteText(ex.ToString());

    public static void WriteText(string text)
    {
        try
        {
            Directory.CreateDirectory(LogDir);
            File.AppendAllText(Path.Combine(LogDir, "depth_client.log"), $"[{DateTime.Now:yyyy-MM-dd HH:mm:ss}]\r\n{text}\r\n\r\n");
        }
        catch
        {
            // Last-resort logging must never throw.
        }
    }
}
