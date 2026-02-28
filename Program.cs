using System;
using System.Drawing;
using System.IO;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Windows.Forms;
using LibVLCSharp.Shared;
using LibVLCSharp.WinForms;
using System.Diagnostics;

namespace zwexecute
{
    public class MainForm : Form
    {
        private LibVLC libVLC;
        private MediaPlayer mediaPlayer;
        private VideoView videoView;
        private bool isPlaying;
        private uint exStyle;
        private string tempFilePath;

        [DllImport("user32.dll")]
        private static extern IntPtr SetWindowLong(IntPtr hWnd, int nIndex, uint dwNewLong);

        [DllImport("user32.dll")]
        private static extern uint GetWindowLong(IntPtr hWnd, int nIndex);

        [DllImport("user32.dll")]
        private static extern bool SetWindowPos(IntPtr hWnd, IntPtr hWndInsertAfter, int X, int Y, int cx, int cy, uint uFlags);

        private const int GWL_EXSTYLE = -20;
        private const uint WS_EX_LAYERED = 0x00080000;
        private const uint WS_EX_TRANSPARENT = 0x00000020;
        private static readonly IntPtr HWND_BOTTOM = new IntPtr(1);
        private const uint SWP_NOSIZE = 0x0001;
        private const uint SWP_NOMOVE = 0x0002;

        public MainForm()
        {
            try
            {
                Core.Initialize();

                videoView = new VideoView
                {
                    Dock = DockStyle.Fill
                };
                this.Controls.Add(videoView);

                libVLC = new LibVLC();
                mediaPlayer = new MediaPlayer(libVLC);
                videoView.MediaPlayer = mediaPlayer;

                this.FormBorderStyle = FormBorderStyle.None;
                this.ShowInTaskbar = false;

                exStyle = GetWindowLong(this.Handle, GWL_EXSTYLE);
                SetWindowLong(this.Handle, GWL_EXSTYLE, exStyle | WS_EX_LAYERED | WS_EX_TRANSPARENT);

                this.Size = Screen.PrimaryScreen.Bounds.Size;
                this.Location = new Point(
                    (Screen.PrimaryScreen.Bounds.Width - this.Width) / 2,
                    (Screen.PrimaryScreen.Bounds.Height - this.Height) / 2
                );

                SetWindowPos(this.Handle, HWND_BOTTOM, 0, 0, 0, 0, SWP_NOSIZE | SWP_NOMOVE);

                mediaPlayer.AspectRatio = null;
                mediaPlayer.Scale = 1.001f;

                StartPlayback();
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"MainForm 初始化出错: {ex.Message}");
            }
        }

        private void StartPlayback()
        {
            try
            {
                if (!string.IsNullOrEmpty(tempFilePath) && File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                }

                string resourceName = "zwexecute.spwj.mp4";
                tempFilePath = Path.Combine(Path.GetTempPath(), "spwj.mp4");

                using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(resourceName))
                {
                    if (stream == null)
                    {
                        Trace.WriteLine($"视频文件未找到: {resourceName}");
                        return;
                    }

                    using (FileStream fileStream = new FileStream(tempFilePath, FileMode.Create, FileAccess.Write))
                    {
                        stream.CopyTo(fileStream);
                    }
                }

                if (mediaPlayer != null && mediaPlayer.IsPlaying)
                {
                    mediaPlayer.EndReached -= OnMediaPlayerEndReached;
                    mediaPlayer.Stop();
                }

                Media media = new Media(libVLC, tempFilePath, FromType.FromPath);
                mediaPlayer.Play(media);
                isPlaying = true;

                mediaPlayer.EndReached += OnMediaPlayerEndReached;
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"播放视频时出错: {ex.Message}");
            }
        }

        private void OnMediaPlayerEndReached(object sender, EventArgs e)
        {
            if (InvokeRequired)
            {
                BeginInvoke(new System.Action(() => OnMediaPlayerEndReached(sender, e)));
                return;
            }

            isPlaying = false;
            mediaPlayer.Stop();
            StartPlayback(); // 直接循环播放
        }

        protected override void OnFormClosing(FormClosingEventArgs e)
        {
            try
            {
                if (mediaPlayer != null)
                {
                    mediaPlayer.EndReached -= OnMediaPlayerEndReached;
                    mediaPlayer.Stop();
                    mediaPlayer.Dispose();
                    mediaPlayer = null;
                }

                if (libVLC != null)
                {
                    libVLC.Dispose();
                    libVLC = null;
                }

                if (!string.IsNullOrEmpty(tempFilePath) && File.Exists(tempFilePath))
                {
                    File.Delete(tempFilePath);
                }
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"关闭窗体时释放资源出错: {ex.Message}");
            }

            e.Cancel = true;
            this.Hide();
        }

        protected override bool ProcessCmdKey(ref Message msg, Keys keyData)
        {
            if (keyData == (Keys.Control | Keys.Shift | Keys.H))
            {
                this.Show();

                if (libVLC == null)
                {
                    Core.Initialize();
                    libVLC = new LibVLC();
                }
                if (mediaPlayer == null)
                {
                    mediaPlayer = new MediaPlayer(libVLC);
                    videoView.MediaPlayer = mediaPlayer;
                }

                return true;
            }
            return base.ProcessCmdKey(ref msg, keyData);
        }

        [STAThread]
        static void Main()
        {
            try
            {
                System.Windows.Forms.Application.EnableVisualStyles();
                System.Windows.Forms.Application.SetCompatibleTextRenderingDefault(false);
                System.Windows.Forms.Application.Run(new MainForm());
            }
            catch (Exception ex)
            {
                Trace.WriteLine($"应用程序启动出错: {ex.Message}");
            }
        }
    }
}
